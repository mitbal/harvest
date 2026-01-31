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


@st.cache_data(ttl=60*10, show_spinner='Downloading dividend data')
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

    cp_df = hd.get_company_profile(final_df['stock'].to_list())
    final_df.drop(columns=['price'], inplace=True)
    final_df = final_df.merge(cp_df[['price', 'changes']], left_on='stock', right_on='symbol')

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

        # start_date = (datetime.today() - timedelta(days=365*10)).isoformat()
        start_date = '2010-01-01'
        price_df = hd.get_daily_stock_price(stock_name, start_from=start_date)
        progress_bar.progress(60, text='Downloading historical price data... Progresss 60%')
        
        if sl == 'JKSE':
            sdf = hd.get_dividend_history_single_stock(stock_name, source='dag')
        else:
            sdf = hd.get_dividend_history_single_stock(stock_name, source='fmp')
        progress_bar.progress(80, text='Downloading historical dividend data... Progresss 80%')

        # sector_df, industry_df = hd.get_sector_industry_pe((date.today()-timedelta(days=2)).isoformat(), api_key)
        progress_bar.progress(100, text='Progress 100% complete')

        end_time = time.time()
        logger.info(f'Total download time for {stock_name}: {end_time-start_time:.04f}')

        time.sleep(0.2)
        progress_bar.empty()

        return fin, cp_df, price_df, sdf, n_share

    except:
        logger.error(f'Error in downloading data for {stock_name}')
        st.error(f'Cannot find the stock {stock_name}. Please check the stock name again and dont forget to add .JK for Indonesian stocks', icon="🚨")
        progress_bar.empty()
        st.stop()


@st.cache_data
def calculate_stock_ratings(stock_name, filtered_df, final_df=None):
    if stock_name in filtered_df.index:
        stock_data = filtered_df.loc[stock_name]
    elif final_df is not None and stock_name in final_df.index:
        stock_data = final_df.loc[stock_name]
    else:
        return None
    
    # 1. Valuation Rating (inverted PE, lower is better)
    # Filter for positive PE only for ranking
    positive_pe_mask = filtered_df['peRatio'] > 0
    
    # Create a series initialized with 0 for all (punish negative PE)
    val_score_series = pd.Series(0.0, index=filtered_df.index)
    
    if positive_pe_mask.any():
        # Rank only positive PEs. ascending=False means High PE gets Low Rank/Pct, Low PE gets High Rank/Pct.
        positive_ranks = filtered_df.loc[positive_pe_mask, 'peRatio'].rank(pct=True, ascending=False)
        val_score_series.loc[positive_pe_mask] = positive_ranks * 100

    if stock_name in val_score_series.index:
        val_score = val_score_series[stock_name]
    else:
        # Fallback: Calculate rank manually against filtered_df
        if stock_data['peRatio'] > 0:
             passing = filtered_df[filtered_df['peRatio'] > 0]
             # Lower PE is better
             if not passing.empty:
                 better_than = passing[passing['peRatio'] > stock_data['peRatio']].shape[0]
                 val_score = better_than / len(passing) * 100
             else:
                 val_score = 0
        else:
             val_score = 0
    
    # 2. Dividend Rating (Consistency & Yield)
    # Average of dividend years percentile and yield percentile
    div_year_rank = filtered_df['numDividendYear'].rank(pct=True)
    yield_rank = filtered_df['yield'].rank(pct=True)
    
    if stock_name in div_year_rank.index:
        div_score = (div_year_rank[stock_name] + yield_rank[stock_name]) / 2 * 100
    else:
        dy_rank = (filtered_df['numDividendYear'] < stock_data['numDividendYear']).mean()
        y_rank = (filtered_df['yield'] < stock_data['yield']).mean()
        div_score = (dy_rank + y_rank) / 2 * 100
    
    # 3. Growth Rating (Revenue & Net Income)
    rev_growth_rank = filtered_df['revenueGrowth'].rank(pct=True)
    inc_growth_rank = filtered_df['netIncomeGrowth'].rank(pct=True)
    
    if stock_name in rev_growth_rank.index:
        growth_score = (rev_growth_rank[stock_name] + inc_growth_rank[stock_name]) / 2 * 100
    else:
        rg_rank = (filtered_df['revenueGrowth'] < stock_data['revenueGrowth']).mean()
        ig_rank = (filtered_df['netIncomeGrowth'] < stock_data['netIncomeGrowth']).mean()
        growth_score = (rg_rank + ig_rank) / 2 * 100
    
    # 4. Profitability Rating (Margin)
    margin_rank = filtered_df['medianProfitMargin'].rank(pct=True)
    if stock_name in margin_rank.index:
        profit_score = margin_rank[stock_name] * 100
    else:
        profit_score = (filtered_df['medianProfitMargin'] < stock_data['medianProfitMargin']).mean() * 100
    
    # 5. Sector & Industry Rating
    # Compare PE vs Sector PE (lower is better relative to sector)
    sector = stock_data['sector']
    sector_df = filtered_df[filtered_df['sector'] == sector]
    if len(sector_df) > 1:
        # Same logic: punish negative PE
        sector_pos_mask = sector_df['peRatio'] > 0
        sector_scores = pd.Series(0.0, index=sector_df.index)
        
        if sector_pos_mask.any():
            sector_ranks = sector_df.loc[sector_pos_mask, 'peRatio'].rank(pct=True, ascending=False)
            sector_scores.loc[sector_pos_mask] = sector_ranks * 100
            
        if stock_name in sector_scores.index:
            sector_score = sector_scores[stock_name]
        else:
             pass_sector = sector_df[sector_df['peRatio'] > 0]
             if not pass_sector.empty and stock_data['peRatio'] > 0:
                better = pass_sector[pass_sector['peRatio'] > stock_data['peRatio']].shape[0]
                sector_score = better / len(pass_sector) * 100
             else:
                sector_score = 0
    else:
        sector_score = 50 # Default middle if no comparison
        
    # 6. Overall Rating
    overall_score = np.mean([val_score, div_score, growth_score, profit_score, sector_score])
    
    return {
        'overall': overall_score,
        'valuation': val_score,
        'dividend': div_score,
        'growth': growth_score,
        'profitability': profit_score,
        'sector': sector_score,
        'metrics': {
            'pe': stock_data['peRatio'],
            'yield': stock_data['yield'],
            'div_years': stock_data['numDividendYear'],
            'rev_growth': stock_data['revenueGrowth'],
            'net_growth': stock_data['netIncomeGrowth'],
            'margin': stock_data['medianProfitMargin'],
            'sector': sector
        }
    }

def get_rating_color(score):
    if score >= 80:
        return '#1b5e20' # Dark Green
    elif score >= 60:
        return '#2e7d32' # Medium Green
    elif score >= 40:
        return '#e37400' # Dark Yellow/Amber
    elif score >= 20: 
        return '#e65100' # Dark Orange
    else:
        return '#c62828' # Dark Red

def render_rating_card(title, score, metrics_dict, chart=None, color=None, key=None):
    if color is None:
        color = get_rating_color(score)
        
    st.markdown(f"### {title}")
    st.markdown(f"## <span style='color:{color}'>{score:.0f}/100</span>", unsafe_allow_html=True)
    
    for label, value in metrics_dict.items():
        if isinstance(value, float):
            st.write(f"**{label}**: {value:.2f}")
        else:
            st.write(f"**{label}**: {value}")
            
    if chart:
        st.altair_chart(chart, use_container_width=True, key=key)

def render_dashboard_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, final_df):
    ratings = calculate_stock_ratings(stock_name, filtered_df, final_df)
    
    if ratings is None:
        st.warning(f"Dashboard View is not available for {stock_name} because it is missing some pre-computed metrics required for the dashboard. Please use correct spelling or switch to Classic View.")
        return

    metrics = ratings['metrics']
    
    # Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    
    card_height = 450
    with r1c1.container(border=True, height=card_height):
        st.markdown(f"### Overall Rating: {stock_name}")
        score = ratings['overall']
        color = get_rating_color(score)
        st.markdown(f"# <span style='color:{color}'>{score:.0f}/100</span>", unsafe_allow_html=True)
        # st.write("Based on average of 5 sub-ratings")
        
        categories = ['Valuation', 'Dividend', 'Growth', 'Profitability', 'Sector']
        data = [ratings['valuation'], ratings['dividend'], ratings['growth'], ratings['profitability'], ratings['sector']]
        radar_option = hp.plot_radar_chart(categories, data, color=color)
        st_echarts(radar_option, height='250px')
        
    with r1c2.container(border=True, height=card_height):
        color = get_rating_color(ratings['valuation'])
        dist_chart = hp.plot_card_distribution(filtered_df, 'peRatio', metrics['pe'], color=color)
        render_rating_card('Valuation Rating', ratings['valuation'], {
            'Current PE': metrics['pe'],
            'Assessment': 'Better than {:.0f}% of stocks'.format(ratings['valuation'])
        }, chart=dist_chart, color=color, key=f"chart_val_{stock_name}")
        
    with r1c3.container(border=True, height=card_height):
        color = get_rating_color(ratings['dividend'])
        yield_chart = hp.plot_card_distribution(filtered_df, 'yield', metrics['yield'], color=color, height=80)
        years_chart = hp.plot_card_histogram(filtered_df, 'numDividendYear', metrics['div_years'], color=color, height=80)
        dist_chart = (yield_chart & years_chart)
        render_rating_card('Dividend Rating', ratings['dividend'], {
            'Yield': f"{metrics['yield']:.2f}% | **Years Paid**: {metrics['div_years']:.0f}",
             'Assessment': 'Better than {:.0f}% of stocks'.format(ratings['dividend'])
        }, chart=dist_chart, color=color, key=f"chart_div_{stock_name}")
        
    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    
    with r2c1.container(border=True, height=card_height):
        color = get_rating_color(ratings['sector'])
        sector_df = filtered_df[filtered_df['sector'] == metrics['sector']]
        if len(sector_df) > 5:
            dist_chart = hp.plot_card_distribution(sector_df, 'peRatio', metrics['pe'], color=color)
        else:
            dist_chart = None
            
        render_rating_card('Sector Rating', ratings['sector'], {
            'Sector': metrics['sector'],
            'In Sector Rank': 'Better than {:.0f}% of peers'.format(ratings['sector'])
        }, chart=dist_chart, color=color, key=f"chart_sect_{stock_name}")
        
    with r2c2.container(border=True, height=card_height):
        color = get_rating_color(ratings['growth'])
        rev_chart = hp.plot_card_distribution(filtered_df, 'revenueGrowth', metrics['rev_growth'], color=color, height=80)
        inc_chart = hp.plot_card_distribution(filtered_df, 'netIncomeGrowth', metrics['net_growth'], color=color, height=80)
        dist_chart = (rev_chart & inc_chart)
        render_rating_card('Growth Rating', ratings['growth'], {
            'Rev Growth': f"{metrics['rev_growth']:.2f}%",
            'Net Inc Growth': f"{metrics['net_growth']:.2f}%"
        }, chart=dist_chart, color=color, key=f"chart_growth_{stock_name}")
        
    with r2c3.container(border=True, height=card_height):
        color = get_rating_color(ratings['profitability'])
        dist_chart = hp.plot_card_distribution(filtered_df, 'medianProfitMargin', metrics['margin'], color=color)
        render_rating_card('Profitability Rating', ratings['profitability'], {
            'Net Margin': f"{metrics['margin']:.2f}%",
            'Assessment': 'Better than {:.0f}% of stocks'.format(ratings['profitability'])
        }, chart=dist_chart, color=color, key=f"chart_profit_{stock_name}")

def render_company_profile(cp_df, stock_name):
    st.write(cp_df.loc[stock_name, 'description'])
    
def render_dividend_history(sdf, final_df, stock_name, filtered_df):
    if sdf is not None:
        dividend_history_cols = st.columns([3, 10, 4])
        dividend_history_cols[0].dataframe(
            sdf[['date', 'adjDividend']],
            column_config={
                'date': st.column_config.DateColumn('Ex-Date'),
                'adjDividend': st.column_config.NumberColumn('Dividend', format='%.01f'),
            },
            hide_index=True)

        try:
            last_val = final_df.loc[stock_name, 'lastDiv']
            inc_val = final_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
        except:
            last_val = 0
            inc_val = 0

        yearly_dividend_chart = hp.plot_dividend_history(sdf, extrapolote=True, n_future_years=5, last_val=last_val, inc_val=inc_val)
        dividend_history_cols[1].altair_chart(yearly_dividend_chart, use_container_width=True)

        with dividend_history_cols[2]:
            if stock_name in filtered_df.index:
                last_div = filtered_df.loc[stock_name, 'lastDiv']
                inc_val = filtered_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
                curr_price = filtered_df.loc[stock_name, 'price']
                next_div = last_div + inc_val
                next_yield = next_div / curr_price * 100

                stats = hd.calc_div_stats(hd.preprocess_div(sdf))
                
                div_years = stats['num_dividend_year']
                pos_years = stats['num_positive_year']
                consistency = pos_years/div_years*100 if div_years > 0 else 0

                dividend_markdown = f'''
                Estimated next year dividend payment: **:green[{next_div:0.2f}]**\n
                Yield on current price: **:green[{next_yield:0.2f}%]**

                Number of years paying dividend: **{div_years:,}**

                Number of years increasing dividend: **{pos_years:,}**

                Positive consistency rate: **:green[{consistency:.2f}%]**
                '''
                st.markdown(dividend_markdown)
    else:
        st.write('No dividend history available')

def render_financial_info(fin, currency, stock_name, filtered_df):
    fin_cols = st.columns([0.3, 0.4, 0.3])
    period = fin_cols[0].radio('Select Period', ['quarter', 'annual'], horizontal=True, index=1, key=f"period_{stock_name}")
    
    if period == 'quarter':
        metric = fin_cols[1].radio('Select Metrics', ['revenue', 'netIncome'], horizontal=True, key=f"metric_{stock_name}")
        fin_chart = hp.plot_financial(fin, period=period, metric=metric, currency=currency)
        with st.container(height=500):
            st.altair_chart(fin_chart, use_container_width=False)
    else:
        fin_view = fin_cols[1].radio('Select View', ['Separate', 'Combined'], horizontal=True, index=1, key=f"view_{stock_name}")
        if fin_view == 'Separate':
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
        else:
            annual_cols = st.columns([80, 20])
            annual_cols[0].write('Annual Financial Chart')
            fin_chart = hp.plot_fin_chart(fin)
            annual_cols[0].altair_chart(fin_chart, use_container_width=True)
            with annual_cols[1]:
                if stock_name in filtered_df.index:
                    st.write('**Financial Metrics Summary**')
                    st.write(f'Average Revenue Growth: {filtered_df.loc[stock_name, "revenueGrowth"]:.2f}%')
                    st.write(f'Average Net Income Growth: {filtered_df.loc[stock_name, "netIncomeGrowth"]:.2f}%')
                    st.write(f'Median Net Profit Margin: {filtered_df.loc[stock_name, "medianProfitMargin"]:.2f}%')

def render_price_movement(price_df):
    candlestick_chart = hp.plot_candlestick(price_df, width=1000, height=300)
    st.altair_chart(candlestick_chart, use_container_width=True)

def render_valuation_analysis(price_df, fin, n_share, sl, stock_name, filtered_df):
    cols = st.columns(3, gap='large')
    year = cols[0].slider('Select Number of Year', min_value=1, max_value=15, key=f"val_year_{stock_name}")
    val_metric = cols[1].radio('Valuation Metric', ['Price-to-Earnings', 'Price-to-Sales/Revenue'], index=0, horizontal=True, key=f"val_metric_{stock_name}")

    val_cols = st.columns(3, gap='large')
    start_date = datetime.now() - timedelta(days=365*year)
    last_year_df = price_df[price_df['date']>= str(start_date)]
    fin_currency = fin.loc[0, 'reportedCurrency']
    target_currency = 'IDR' if sl == 'JKSE' else 'USD'
    
    if val_metric == 'Price-to-Earnings':
        ratio = 'P/E'; pratio = 'peRatio'
        pe_df = hd.calc_ratio_history(last_year_df, fin, n_shares=n_share, ratio='pe', reported_currency=fin_currency, target_currency=target_currency)
    else:
        ratio = 'P/S'; pratio = 'psRatio'
        pe_df = hd.calc_ratio_history(last_year_df, fin, n_shares=n_share, ratio='ps', reported_currency=fin_currency, target_currency=target_currency)
        
    if stock_name in filtered_df.index:
        sector_name = filtered_df.loc[stock_name, 'sector']
        industry_name = filtered_df.loc[stock_name, 'industry']
        sector_df = filtered_df[filtered_df['sector'] == sector_name]
        sector_pe = (sector_df['mktCap'] * sector_df[pratio]).sum() / sector_df['mktCap'].sum()
        industry_df = filtered_df[filtered_df['industry'] == industry_name]
        industry_pe = (industry_df['mktCap'] * industry_df[pratio]).sum() / industry_df['mktCap'].sum()
    else:
        sector_pe = industry_pe = -1

    pe_ttm = pe_df['pe'].values[-1]
    current_price = price_df['close'].values[0]
    median_pe = pe_df['pe'].median()
    
    pe_dist_chart = hp.plot_pe_distribution(pe_df, pe_ttm, axis_label=ratio)
    val_cols[0].altair_chart(pe_dist_chart, use_container_width=True)
    pe_ts_chart = hp.plot_pe_timeseries(pe_df, axis_label=ratio)
    val_cols[1].altair_chart(pe_ts_chart, use_container_width=True)

    highlight_color = 'green' if pe_ttm <= median_pe else 'red'
    sector_color = 'green' if pe_ttm <= sector_pe else 'red'
    industry_color = 'green' if pe_ttm <= industry_pe else 'red'

    with val_cols[2]:
        ci = pe_df['pe'].quantile([.05, .95]).values
        markdown_table = f"""
        | Metric | Value |
        | ------ | ----- |
        | Current {ratio} | **:{highlight_color}[{pe_ttm:.2f}]** |
        | Current Price | **:{highlight_color}[{int(current_price):,}]** |
        | Median last {year} year {ratio} | {median_pe:.2f} |
        | Fair Price | {int((median_pe/pe_ttm)*current_price):,} |
        | 95% Confidence Interval range {ratio} | {ci[0]:.2f} - {ci[1]:.2f} |
        | 95% Confidence Interval range Price | {int((ci[0]/pe_ttm)*current_price):,} - {int((ci[1]/pe_ttm)*current_price):,} |
        """
        if industry_pe != -1 and sector_pe != -1:
            markdown_table += f"| Industry: {industry_name} {ratio} | **:{industry_color}[{industry_pe:.2f}]** | \n \
        | Sector: {sector_name} {ratio} | **:{sector_color}[{sector_pe:.2f}]** |"
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

def render_compounding_simulation(stock_name, price_df, sdf):
    this_year = datetime.now().year
    cols = st.columns(2)
    start_year = cols[0].number_input(label='Start Year', value=2021, min_value=2010, max_value=this_year-2, key=f"sim_start_{stock_name}")
    end_year = cols[1].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1, key=f"sim_end_{stock_name}")
    cols = st.columns(2)
    initial_value = cols[0].number_input(label='Initial investment (in million)', value=10, min_value=1, max_value=1000, key=f"sim_init_{stock_name}")
    monthly_topup = cols[1].number_input(label='Monthly Topup (in million)', value=1, min_value=0, max_value=100, key=f"sim_monthly_{stock_name}")

    porto, activities = hd.simulate_dividend_compounding(stock_name, price_df, sdf, start_year, end_year, initial_value * 1_000_000, monthly_topup * 1_000_000)
    
    st.write('Activities:')
    st.write(activities)
    st.write('Portfolio:')

    porto_df = pd.DataFrame(porto)
    porto_df['total'] = porto_df['num_stock'] * porto_df['price'] * 100
    porto_df['cum_stock'] = porto_df['num_stock'].cumsum()
    porto_df['cum_total'] = porto_df['total'].cumsum()
    if not porto_df['cum_stock'].empty and (porto_df['cum_stock'] != 0).all():
         porto_df['avg_price'] = porto_df['cum_total'] / porto_df['cum_stock'] / 100
    else:
         porto_df['avg_price'] = 0

    porto_df['current_value'] = porto_df['price'] * porto_df['cum_stock'] * 100

    cols = st.columns(2)
    cols[0].write(porto_df[['date', 'cum_total', 'current_value']])

    return_chart1 = alt.Chart(porto_df).mark_line().encode(x=alt.X('date', axis=alt.Axis(labels=False)), y=alt.Y('cum_total').scale(zero=False))
    return_chart2 = alt.Chart(porto_df).mark_line().encode(x=alt.X('date', axis=alt.Axis(labels=False)), y=alt.Y('current_value').scale(zero=False), color=alt.value('green'))
    cols[1].altair_chart(return_chart1 + return_chart2)

def render_classic_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, sl):
    with st.expander('Company Profile', expanded=False):    
        render_company_profile(cp_df, stock_name)
    
    currency = cp_df.loc[stock_name, 'currency']

    with st.expander(f'Dividend History: {stock_name}', expanded=True):
        render_dividend_history(sdf, filtered_df, stock_name, filtered_df)
        
    with st.expander(f'Financial Information: {stock_name}', expanded=True):
        render_financial_info(fin, currency, stock_name, filtered_df)
        
    with st.expander('Price Movement', expanded=True):
        render_price_movement(price_df)
        
    with st.expander(f'Valuation Analysis: {stock_name}', expanded=True):
        render_valuation_analysis(price_df, fin, n_share, sl, stock_name, filtered_df)

    with st.expander(f'Compounding Simulation: {stock_name}'):
        render_compounding_simulation(stock_name, price_df, sdf)


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
if sl != 'JKSE':
    final_df = final_df.drop('GOOGL')

if sl == 'JKSE':
    is_syariah = st.sidebar.toggle('Syariah Only?')
    if is_syariah:
        final_df = final_df[final_df['is_syariah'] == True]

default_view = 'Table'
if 'view' in st.query_params:
    if st.query_params['view'] == 'treemap':
        default_view = 'Treemap'

full_table_section = st.container(border=True)
with full_table_section:

    # final_df['DScore'] = final_df['DScore'] * (10/final_df['peRatio']**3) * final_df['medianProfitMargin']
    final_df['marginTTM'] = final_df['earningTTM'] / final_df['revenueTTM'] * 100
    # final_df['revenueGrowthTTM'] = final_df['revenueGrowthTTM'] * 100
    # final_df['netIncomeGrowthTTM'] = final_df['netIncomeGrowthTTM'] * 100
    # final_df['PEG TTM'] = final_df['peRatio'] / final_df['netIncomeGrowthTTM']
    # final_df['PEG Median'] = final_df['peRatio'] / final_df['netIncomeGrowth']

    filtered_df = final_df[(final_df['mktCap'] >= minimum_market_cap*1000_000_000)
                            & (final_df['numDividendYear'] >= minimum_year)
                            & (final_df['lastDiv'] > 0)].sort_values('DScore', ascending=False)

    view = st.segmented_control(label='View Option', 
                         options=['Table', 'Treemap', 'Scatter Plot', 'Distribution'],
                         selection_mode='single',
                         default=default_view)

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
            'earningTTM': st.column_config.NumberColumn(
                'Earning TTM',
                help='Earning in the last twelve months',
                format='localized'
            ),
            'revenueTTM': st.column_config.NumberColumn(
                'Revenue TTM',
                help='Revenue in the last twelve months',
                format='localized'
            ),
            'peRatio': st.column_config.NumberColumn(
                'PE Ratio',
                help='Price to Earnings Ratio',
                format='%.2f'
            ),
            'psRatio': st.column_config.NumberColumn(
                'PS Ratio',
                help='Price to Sales/Revenue Ratio',
                format='%.2f'
            )

        }

        event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun', column_config=cfig)

    elif view == 'Treemap':
        
        treemap_cols = st.columns([1,1,3])
        size_var = treemap_cols[0].selectbox(options=['Market Cap', 'Revenue', 'Net Income', 'Dividend Yield'], label='Select Size Variable')
        color_var = treemap_cols[1].selectbox(options=['None', 'Dividend Yield', 'Median Profit Margin', 'TTM Profit Margin', 'Revenue Growth', 'Daily Return', 'PE Ratio', 'PS Ratio'], label='Select Color Variable', index=1)
        sector_var = treemap_cols[2].selectbox(options=['ALL']+filtered_df['sector'].unique().tolist(), label='Select Sector')
        
        df_tree = filtered_df[['sector', 'industry']].copy()

        df_tree.loc[:, 'Market Cap'] = filtered_df['mktCap'] / 1_000_000_000
        df_tree.loc[:, 'Revenue'] = filtered_df['revenueTTM']
        df_tree.loc[:, 'Net Income'] = filtered_df['earningTTM']
        df_tree.loc[:, 'Dividend Yield'] = filtered_df['yield']
        df_tree.loc[:, 'Median Profit Margin'] = filtered_df['medianProfitMargin']
        df_tree.loc[:, 'TTM Profit Margin'] = filtered_df['marginTTM']
        df_tree.loc[:, 'Revenue Growth'] = filtered_df['revenueGrowth']
        df_tree.loc[:, 'Daily Return'] = filtered_df['changes'] / filtered_df['price'] * 100
        df_tree.loc[:, 'PE Ratio'] = filtered_df['peRatio']
        df_tree.loc[:, 'PS Ratio'] = filtered_df['psRatio']
        df_tree = df_tree.dropna()

        if sector_var != 'ALL':
            df_tree = df_tree[df_tree['sector'] == sector_var]

        if color_var == 'None':
            color_var = None
            show_gradient = False
            add_label = None
        else:
            show_gradient = True
            add_label = 'color_var'

        color_map = 'green_shade'
        color_threshold = None
        if color_var == 'Daily Return':
            color_map = 'red_green'
            color_threshold = [-3, -1, 0, 1, 3]
        elif color_var == 'PE Ratio':
            color_map = 'red_shade'
            color_threshold = [-100, 0, 5, 15]
        elif color_var == 'PS Ratio':
            color_map = 'red_shade'
            color_threshold = [-1000, -100, -10, -1, 0, 1, 2, 3, 5]

        tree_data = hd.prep_treemap(df_tree, size_var=size_var, color_var=color_var, color_threshold=color_threshold, add_label=add_label)
        option = hp.plot_treemap(tree_data, size_var=size_var, color_var=color_var, show_gradient=show_gradient, colormap=color_map)
        
        click_event_js = """function(params){console.log('Clicked item:',params.name);return params.name;}"""
        
        clicked_item_name = st_echarts(
            option,
            events={"click": click_event_js},
            height='900px', 
            width='1550px'
        )
    
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
elif view == 'Treemap' and clicked_item_name:
    stock_name = clicked_item_name.split()[0]
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

fin, cp_df, price_df, sdf, n_share = get_specific_stock_detail(stock_name)


# stock_view_mode = st.radio("View Mode", ["Classic", "Dashboard"], horizontal=True)
stock_view_mode = st.toggle("Dashboard View", value=False)
mode = "Dashboard" if stock_view_mode else "Classic"

if mode == "Classic":
    render_classic_view(stock_name, final_df, fin, cp_df, price_df, sdf, n_share, sl)
else:
    render_dashboard_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, final_df)
