import os
import io
import json
import time
import psutil
import logging
import concurrent.futures
from pythonjsonlogger.jsonlogger import JsonFormatter

import redis
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_echarts5 import st_echarts
from datetime import date, datetime, timedelta

import harvest.plot as hp
import harvest.data as hd
from harvest.utils import setup_logging


st.title('Jajan Saham')

api_key = os.environ['FMP_API_KEY']
redis_url = os.environ['REDIS_URL']

# Constants for Dividend Score (DScore) calculation
DIV_MATURITY_HALFLIFE = 25   # Years until dividend consistency is considered "mature"
PROJECTION_HORIZON_YRS = 5   # Number of forecast years for dividend extrapolation

### Start of Function definition


@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
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
        if isinstance(rjson, bytes) and rjson.startswith(b'PAR1'):
            final_df = pd.read_parquet(io.BytesIO(rjson))
            logger.info("dividend table loaded from parquet")
        else:
            div_score_json = json.loads(rjson)
            if 'date' in div_score_json:
                last_updated = div_score_json['date']
                logger.info(f'dividend table last updated: {last_updated}')
                final_df = pd.DataFrame(json.loads(div_score_json['content']))
            else:
                final_df = pd.DataFrame(div_score_json)
    else:
        final_df = pd.read_csv('dividend_historical.csv')

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)

    cp_df = hd.get_company_profile(final_df['stock'].to_list())
    final_df.drop(columns=['price'], inplace=True)
    final_df = final_df.merge(cp_df[['price', 'changes', 'beta']], left_on='stock', right_on='symbol')

    return final_df.set_index('stock')


@st.cache_data(ttl=60*60, show_spinner=False)
def get_specific_stock_detail(stock_name, sl):
    """Pure data-fetching function — no UI side-effects so cache is safe to share."""
    start_time = time.time()

    n_share = hd.get_shares_outstanding(stock_name)['outstandingShares'].tolist()[0]
    fin     = hd.get_financial_data(stock_name)
    cp_df   = hd.get_company_profile([stock_name])

    start_date = '2010-01-01'
    price_df   = hd.get_daily_stock_price(stock_name, start_from=start_date)

    source = 'dag' if sl == 'JKSE' else 'fmp'
    sdf    = hd.get_dividend_history_single_stock(stock_name, source=source)

    end_time = time.time()
    logger.info(f'Total download time for {stock_name}: {end_time-start_time:.04f}')

    return fin, cp_df, price_df, sdf, n_share


def calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share):
    """
    Calculate metrics for a stock that is not in the pre-computed table.
    """
    stats = {}
    
    # 1. Basic Info from Company Profile
    if stock_name in cp_df.index:
        cp = cp_df.loc[stock_name]
        stats['price'] = cp.get('price', 0)
        stats['changes'] = cp.get('changes', 0)
        stats['mktCap'] = cp.get('mktCap', 0)
        stats['sector'] = cp.get('sector', 'Unknown')
        stats['industry'] = cp.get('industry', 'Unknown')
        stats['beta'] = cp.get('beta', 1.0)
        
        # Calculate years since IPO if available
        ipo_date = cp.get('ipoDate')
        if ipo_date:
            try:
                ipo_dt = datetime.strptime(ipo_date, '%Y-%m-%d')
                stats['numOfYear'] = (datetime.now() - ipo_dt).days / 365.25
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing IPO date '{ipo_date}' for {stock_name}: {e}")
                stats['numOfYear'] = 10  # Default fallback
        else:
            stats['numOfYear'] = 10
    else:
        stats['price'] = price_df['close'].iloc[0] if not price_df.empty else 0
        stats['changes'] = 0
        stats['mktCap'] = stats['price'] * n_share
        stats['sector'] = 'Unknown'
        stats['industry'] = 'Unknown'
        stats['numOfYear'] = 10

    # 2. Dividend Stats
    if sdf is not None and not sdf.empty:
        div_pp = hd.preprocess_div(sdf)
        div_stats = hd.calc_div_stats(div_pp)
        
        stats['yield'] = (sdf['adjDividend'].iloc[0] * 4 / stats['price'] * 100) if stats['price'] > 0 else 0 # Rough estimate of annual yield if quarterly
        # Use last year total for more accurate yield
        if not div_pp.empty:
            last_year_div = div_pp['adjDividend'].iloc[-1]
            stats['yield'] = (last_year_div / stats['price'] * 100) if stats['price'] > 0 else 0
            stats['lastDiv'] = last_year_div
        else:
            stats['lastDiv'] = 0
            
        stats['numDividendYear'] = div_stats.get('num_dividend_year', 0)
        stats['positiveYear'] = div_stats.get('num_positive_year', 0)
        stats['avgFlatAnnualDivIncrease'] = div_stats.get('historical_mean_flat', 0)
    else:
        stats['yield'] = 0
        stats['numDividendYear'] = 0
        stats['positiveYear'] = 0
        stats['avgFlatAnnualDivIncrease'] = 0
        stats['lastDiv'] = 0

    # 3. Financial Stats
    if fin is not None and not fin.empty:
        # Determine target currency based on company profile or stock suffix
        if stock_name in cp_df.index and 'currency' in cp_df.columns:
            target_curr = cp_df.loc[stock_name, 'currency']
        else:
            target_curr = 'IDR' if stock_name.endswith('.JK') else 'USD'
            
        fin_stats = hd.calc_fin_stats(fin, target_currency=target_curr)
        
        stats['revenueGrowth'] = fin_stats.get('median_5y_revenue_growth', 0)
        stats['netIncomeGrowth'] = fin_stats.get('median_5y_netIncome_growth', 0)
        stats['medianProfitMargin'] = fin_stats.get('median_profit_margin', 0)
        stats['earningTTM'] = fin_stats.get('earningTTM', 0)
        stats['revenueTTM'] = fin_stats.get('revenueTTM', 0)
        stats['marginTTM'] = fin_stats.get('marginTTM', 0)
        stats['revenueGrowthTTM'] = fin_stats.get('revenue_growth_TTM', 0)
        stats['netIncomeGrowthTTM'] = fin_stats.get('netIncome_growth_TTM', 0)
        
        if stats['earningTTM'] > 0:
            stats['peRatio'] = stats['mktCap'] / stats['earningTTM']
        else:
            stats['peRatio'] = -1
            
        if stats['revenueTTM'] > 0:
            stats['psRatio'] = stats['mktCap'] / stats['revenueTTM']
        else:
            stats['psRatio'] = -1
    else:
        stats['revenueGrowth'] = 0
        stats['netIncomeGrowth'] = 0
        stats['medianProfitMargin'] = 0
        stats['peRatio'] = -1
        stats['psRatio'] = -1
        stats['earningTTM'] = 0
        stats['revenueTTM'] = 0

    return pd.Series(stats, name=stock_name)


@st.cache_data(show_spinner=False)
def calculate_missing_stats_cached(stock_name, fin_json, cp_json, price_json, sdf_json, n_share):
    """
    Cached wrapper around calculate_missing_stats.
    DataFrames are passed as JSON strings so Streamlit can hash them.
    """
    fin      = pd.read_json(fin_json)      if fin_json      else pd.DataFrame()
    cp_df    = pd.read_json(cp_json)      if cp_json       else pd.DataFrame()
    price_df = pd.read_json(price_json)   if price_json    else pd.DataFrame()
    sdf      = pd.read_json(sdf_json)     if sdf_json      else None
    return calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share)


# Columns from filtered_df that rating logic actually needs — serialise only these as the cache key
_RATING_COLS = ['peRatio', 'numDividendYear', 'yield', 'revenueGrowth', 'netIncomeGrowth', 'medianProfitMargin', 'sector']


@st.cache_data(show_spinner=False)
def _calculate_stock_ratings_cached(stock_name, ranking_df_json, stock_data_json):
    """
    Cached implementation — DataFrames are passed as JSON strings so Streamlit hashes
    them cheaply. Only the narrow columns needed for rating are serialised.
    """
    ranking_df = pd.read_json(io.StringIO(ranking_df_json))
    stock_data = pd.read_json(io.StringIO(stock_data_json), typ='series')

    # 1. Valuation Rating (inverted PE, lower is better)
    positive_pe_mask = ranking_df['peRatio'] > 0
    val_score_series = pd.Series(0.0, index=ranking_df.index)
    if positive_pe_mask.any():
        positive_ranks = ranking_df.loc[positive_pe_mask, 'peRatio'].rank(pct=True, ascending=False)
        val_score_series.loc[positive_pe_mask] = positive_ranks * 100

    if stock_name in val_score_series.index:
        val_score = val_score_series[stock_name]
    else:
        if stock_data['peRatio'] > 0:
            passing = ranking_df[ranking_df['peRatio'] > 0]
            if not passing.empty:
                better_than = passing[passing['peRatio'] > stock_data['peRatio']].shape[0]
                val_score = better_than / len(passing) * 100
            else:
                val_score = 0
        else:
            val_score = 0

    # 2. Dividend Rating (Consistency & Yield)
    div_year_rank = ranking_df['numDividendYear'].rank(pct=True)
    yield_rank    = ranking_df['yield'].rank(pct=True)
    if stock_name in div_year_rank.index:
        div_score = (div_year_rank[stock_name] + yield_rank[stock_name]) / 2 * 100
    else:
        dy_rank   = (ranking_df['numDividendYear'] < stock_data['numDividendYear']).mean()
        y_rank    = (ranking_df['yield'] < stock_data['yield']).mean()
        div_score = (dy_rank + y_rank) / 2 * 100

    # 3. Growth Rating (Revenue & Net Income)
    rev_growth_rank = ranking_df['revenueGrowth'].rank(pct=True)
    inc_growth_rank = ranking_df['netIncomeGrowth'].rank(pct=True)
    if stock_name in rev_growth_rank.index:
        growth_score = (rev_growth_rank[stock_name] + inc_growth_rank[stock_name]) / 2 * 100
    else:
        rg_rank      = (ranking_df['revenueGrowth'] < stock_data['revenueGrowth']).mean()
        ig_rank      = (ranking_df['netIncomeGrowth'] < stock_data['netIncomeGrowth']).mean()
        growth_score = (rg_rank + ig_rank) / 2 * 100

    # 4. Profitability Rating (Margin)
    margin_rank = ranking_df['medianProfitMargin'].rank(pct=True)
    if stock_name in margin_rank.index:
        profit_score = margin_rank[stock_name] * 100
    else:
        profit_score = (ranking_df['medianProfitMargin'] < stock_data['medianProfitMargin']).mean() * 100

    # 5. Sector & Industry Rating
    sector    = stock_data['sector']
    sector_df = ranking_df[ranking_df['sector'] == sector]
    if len(sector_df) > 1:
        sector_pos_mask = sector_df['peRatio'] > 0
        sector_scores   = pd.Series(0.0, index=sector_df.index)
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
        sector_score = 50  # Default middle if no sector peers

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
            'pe':         stock_data['peRatio'],
            'yield':      stock_data['yield'],
            'div_years':  stock_data['numDividendYear'],
            'rev_growth': stock_data['revenueGrowth'],
            'net_growth': stock_data['netIncomeGrowth'],
            'margin':     stock_data['medianProfitMargin'],
            'sector':     sector,
        }
    }


def calculate_stock_ratings(stock_name, filtered_df, final_df=None, stock_data=None):
    """
    Public wrapper: resolves stock_data, narrows DataFrames to only the columns
    needed for rating, serialises them to JSON, then delegates to the cached impl.
    """
    if stock_data is None:
        if stock_name in filtered_df.index:
            stock_data = filtered_df.loc[stock_name]
        elif final_df is not None and stock_name in final_df.index:
            stock_data = final_df.loc[stock_name]
        else:
            return None

    # Build a narrow ranking DataFrame — only the 7 columns the algorithm uses
    available_rating_cols = [c for c in _RATING_COLS if c in filtered_df.columns]
    ranking_df = filtered_df[available_rating_cols]

    # Serialise stock_data as a Series keeping only rating columns
    stock_series = stock_data[available_rating_cols] if hasattr(stock_data, '__getitem__') else stock_data

    return _calculate_stock_ratings_cached(
        stock_name,
        ranking_df.to_json(),
        pd.Series(stock_series).to_json(),
    )

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
            st.write(f"**{label}**: {value:,.2f}")
        else:
            st.write(f"**{label}**: {value}")
            
    if chart:
        st.altair_chart(chart, width='stretch', key=key)

def render_dashboard_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, final_df):
    
    if stock_name in filtered_df.index:
        stock_data = filtered_df.loc[stock_name]
    elif final_df is not None and stock_name in final_df.index:
        stock_data = final_df.loc[stock_name]
    else:
        # Recalculate stats on the fly
        stock_data = calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share)

    ratings = calculate_stock_ratings(stock_name, filtered_df, final_df, stock_data=stock_data)
    
    if ratings is None:
        st.warning(f"Dashboard View could not be generated for {stock_name}. Please try Classic View.")
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
        
        zero_yield_stocks = (filtered_df['yield'] == 0).sum()
        yield_df = filtered_df[filtered_df['yield'] > 0]
        
        yield_chart = hp.plot_card_distribution(yield_df, 'yield', metrics['yield'], color=color, height=80)
        years_chart = hp.plot_card_histogram(yield_df, 'numDividendYear', metrics['div_years'], color=color, height=80)
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
    
def render_dividend_history(sdf, final_df, stock_name, filtered_df, fin=None, n_share=None, currency='IDR', cp_df=None, price_df=None):
    if sdf is not None:
        dividend_history_cols = st.columns([3, 10, 4])
        dividend_history_cols[0].dataframe(
            sdf[['date', 'adjDividend']],
            column_config={
                'date': st.column_config.DateColumn('Ex-Date'),
                'adjDividend': st.column_config.NumberColumn('Dividend', format='%,.1f'),
            },
            hide_index=True)

        try:
            last_val = final_df.loc[stock_name, 'lastDiv']
            inc_val = final_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
        except KeyError as e:
            logger.warning(f"Stock {stock_name} not found in final_df for dividend stats: {e}")
            last_val = 0
            inc_val = 0
        except Exception as e:
            logger.error(f"Unexpected error getting dividend stats for {stock_name}: {e}")
            last_val = 0
            inc_val = 0
            
        annual_eps_df = None
        if fin is not None and not fin.empty and 'calendarYear' in fin.columns and 'netIncome' in fin.columns and n_share is not None:
            f = fin.copy()
            reported_currency = f.loc[0, 'reportedCurrency'] if 'reportedCurrency' in f.columns else currency
            exchange_rate = hd.get_usd_idr_rate() if currency == 'IDR' and reported_currency == 'USD' else 1
            f['year'] = f['calendarYear'].astype(int)
            f = f.groupby('year')['netIncome'].sum().reset_index()
            # Calculate EPS
            f['eps'] = f['netIncome'] * exchange_rate / n_share
            annual_eps_df = f[['year', 'eps']]

        avg_payout_str = "N/A"
        if annual_eps_df is not None:
            sdf_yr = sdf.copy()
            sdf_yr['year'] = sdf_yr['date'].apply(lambda x: int(x.split('-')[0]))
            yearly_div = sdf_yr.groupby('year')['adjDividend'].sum().reset_index()
            payout_df = pd.merge(yearly_div, annual_eps_df, on='year', how='inner')
            
            payout_ratios = []
            for _, row in payout_df.iterrows():
                if pd.notna(row['eps']) and row['eps'] > 0:
                    pr = (row['adjDividend'] / row['eps']) * 100
                    payout_ratios.append(pr)
            if payout_ratios:
                avg_pr = sum(payout_ratios) / len(payout_ratios)
                avg_payout_str = f"**:green[{avg_pr:.2f}%]**" if avg_pr <= 100 else f"**:red[{avg_pr:.2f}%]**"

        yearly_dividend_chart = hp.plot_dividend_history(sdf, extrapolote=True, n_future_years=5, last_val=last_val, inc_val=inc_val)
        dividend_history_cols[1].altair_chart(yearly_dividend_chart, width='stretch')

        with dividend_history_cols[2]:
            if stock_name in filtered_df.index:
                stock_data = filtered_df.loc[stock_name]
            else:
                stock_data = calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share)
                
            last_div = stock_data['lastDiv']
            inc_val = stock_data['avgFlatAnnualDivIncrease']
            curr_price = stock_data['price']
            pe_ratio = stock_data['peRatio']
            
            next_div = last_div + inc_val
            next_yield = next_div / curr_price * 100 if curr_price > 0 else 0

            if pe_ratio and pe_ratio > 0:
                eps = curr_price / pe_ratio if pe_ratio > 0 else 0
                payout_ratio = (last_div / eps) * 100 if eps > 0 else 0
                payout_str = f"**:green[{payout_ratio:.2f}%]**" if payout_ratio <= 100 else f"**:red[{payout_ratio:.2f}%]**"
            else:
                payout_str = "N/A"

            stats = hd.calc_div_stats(hd.preprocess_div(sdf))
            
            div_years = stats['num_dividend_year']
            pos_years = stats['num_positive_year']
            consistency = pos_years/div_years*100 if div_years > 0 else 0

            cagr_5y = stats.get('cagr_5y')
            cagr_10y = stats.get('cagr_10y')
            
            cagr_5y_str = f"**:green[{cagr_5y:.2f}%]**" if cagr_5y is not None and cagr_5y >= 0 else (f"**:red[{cagr_5y:.2f}%]**" if cagr_5y is not None else "N/A")
            cagr_10y_str = f"**:green[{cagr_10y:.2f}%]**" if cagr_10y is not None and cagr_10y >= 0 else (f"**:red[{cagr_10y:.2f}%]**" if cagr_10y is not None else "N/A")

            dividend_markdown = f'''
            Estimated next year dividend payment: **:green[{next_div:,.2f}]**\n
            Yield on current price: **:green[{next_yield:,.2f}%]**\n
            Payout Ratio (Current): {payout_str}\n
            Payout Ratio (Average): {avg_payout_str}

            Number of years paying dividend: **{div_years:,}**

            Number of years increasing dividend: **{pos_years:,}**

            Positive consistency rate: **:green[{consistency:,.2f}%]**

            5-Year CAGR: {cagr_5y_str}

            10-Year CAGR: {cagr_10y_str}
            '''
            st.markdown(dividend_markdown)
    else:
        st.write('No dividend history available')

def render_financial_info(fin, currency, stock_name, filtered_df):

    # Only copy the columns we might mutate — avoids a full 60-row × 50-col copy
    cols_needed = [c for c in ['date', 'calendarYear', 'revenue', 'netIncome', 'reportedCurrency',
                                'period', 'eps'] if c in fin.columns]
    fin = fin[cols_needed].copy()
    if not fin.empty and 'reportedCurrency' in fin.columns:
        reported_currency = fin.loc[0, 'reportedCurrency']
        if currency == 'IDR' and reported_currency == 'USD':
            exchange_rate = hd.get_usd_idr_rate()
            if 'revenue' in fin.columns:
                fin['revenue'] = fin['revenue'] * exchange_rate
            if 'netIncome' in fin.columns:
                fin['netIncome'] = fin['netIncome'] * exchange_rate

    fin_cols = st.columns([0.3, 0.4, 0.3])
    period = fin_cols[0].radio('Select Period', ['quarter', 'annual'], horizontal=True, index=1, key=f"period_{stock_name}")
    
    if period == 'quarter':
        metric = fin_cols[1].radio('Select Metrics', ['revenue', 'netIncome'], horizontal=True, key=f"metric_{stock_name}")
        fin_chart = hp.plot_financial(fin, period=period, metric=metric, currency=currency)
        with st.container(height=500):
            st.altair_chart(fin_chart, width='content')
    else:
        fin_view = fin_cols[1].radio('Select View', ['Separate', 'Combined'], horizontal=True, index=1, key=f"view_{stock_name}")
        if fin_view == 'Separate':
            annual_cols = st.columns([40,40,20])
            annual_cols[0].write('Annual Revenue Chart')
            revenue_chart = hp.plot_financial(fin, period=period, metric='revenue', currency=currency)
            annual_cols[0].altair_chart(revenue_chart, width='stretch')
            annual_cols[1].write('Annual Net Income Chart')
            income_chart = hp.plot_financial(fin, period=period, metric='netIncome', currency=currency)
            annual_cols[1].altair_chart(income_chart, width='stretch')
            with annual_cols[2]:
                if stock_name in filtered_df.index:
                    stock_data = filtered_df.loc[stock_name]
                else:
                    stock_data = calculate_missing_stats(stock_name, fin, pd.DataFrame(), pd.DataFrame(), None, 0) # Minimal call
                fin_stats = cached_calc_fin_stats(fin.to_json(), currency)
                _render_fin_summary(fin_stats, stock_data)
        else:
            annual_cols = st.columns([80, 20])
            annual_cols[0].write('Annual Financial Chart')
            fin_chart = hp.plot_fin_chart(fin, currency=currency)
            annual_cols[0].altair_chart(fin_chart, width='stretch')
            with annual_cols[1]:
                if stock_name in filtered_df.index:
                    stock_data = filtered_df.loc[stock_name]
                else:
                    stock_data = calculate_missing_stats(stock_name, fin, pd.DataFrame(), pd.DataFrame(), None, 0) # Minimal call
                fin_stats = cached_calc_fin_stats(fin.to_json(), currency)
                _render_fin_summary(fin_stats, stock_data)

def _render_fin_metric(label, val, avg):
    """Colour-coded financial metric line: green if better than avg, red if clearly worse."""
    if val > avg:
        st.markdown(f'{label}: **:green[{val:.2f}%]**')
    elif val < avg * 0.9:
        st.markdown(f'{label}: **:red[{val:.2f}%]**')
    else:
        st.write(f'{label}: **{val:.2f}%**')


def _render_fin_summary(fin_stats, stock_data):
    """Render the TTM vs historical financial metrics summary panel."""
    st.write('**Financial Metrics Summary**')
    _render_fin_metric('TTM Revenue Growth',   fin_stats['revenue_growth_TTM'], stock_data['revenueGrowth'])
    _render_fin_metric('TTM Net Income Growth', fin_stats['netIncome_growth_TTM'], stock_data['netIncomeGrowth'])
    _render_fin_metric('TTM Net Profit Margin', fin_stats['marginTTM'],           stock_data['medianProfitMargin'])
    st.write(f'Average Revenue Growth: {stock_data["revenueGrowth"]:.2f}%')
    st.write(f'Average Net Income Growth: {stock_data["netIncomeGrowth"]:.2f}%')
    st.write(f'Median Net Profit Margin: {stock_data["medianProfitMargin"]:.2f}%')
    st.write('---')


def render_price_movement(price_df):
    candlestick_chart = hp.plot_candlestick(price_df, width=1000, height=300)
    st.altair_chart(candlestick_chart, width='content')


@st.cache_data(show_spinner=False)
def cached_calc_ratio_history(price_df_json, fin_json, n_shares, ratio, reported_currency, target_currency):
    """Cached wrapper — roll-over PE/PS ratio history only recomputed when inputs change."""
    price_df = pd.read_json(io.StringIO(price_df_json))
    fin      = pd.read_json(io.StringIO(fin_json))
    return hd.calc_ratio_history(price_df, fin, n_shares=n_shares, ratio=ratio,
                                 reported_currency=reported_currency, target_currency=target_currency)


@st.cache_data(show_spinner=False)
def cached_calc_fin_stats(fin_json, target_currency):
    """Cached wrapper — fin stats recomputed only when the financial data changes."""
    fin = pd.read_json(io.StringIO(fin_json))
    return hd.calc_fin_stats(fin, target_currency=target_currency)

def render_valuation_analysis(price_df, fin, n_share, sl, stock_name, filtered_df, cp_df=None, sdf=None):
    cols = st.columns(3, gap='large')
    year = cols[0].slider('Select Number of Year', min_value=1, max_value=15, key=f"val_year_{stock_name}")
    val_metric = cols[1].radio('Valuation Metric', ['Price-to-Earnings', 'Price-to-Sales/Revenue'], index=0, horizontal=True, key=f"val_metric_{stock_name}")

    val_cols = st.columns(3, gap='large')
    start_date = datetime.now() - timedelta(days=365*year)
    last_year_df = price_df[price_df['date']>= str(start_date)]
    fin_currency = fin.loc[0, 'reportedCurrency']
    target_currency = 'IDR' if sl == 'JKSE' else 'USD'

    # Serialise to JSON once — the cached helpers use these as cache keys
    last_year_json = last_year_df.to_json()
    fin_json_val   = fin.to_json()

    if val_metric == 'Price-to-Earnings':
        ratio = 'P/E'; pratio = 'peRatio'
        pe_df = cached_calc_ratio_history(last_year_json, fin_json_val, n_share, 'pe', fin_currency, target_currency)
    else:
        ratio = 'P/S'; pratio = 'psRatio'
        pe_df = cached_calc_ratio_history(last_year_json, fin_json_val, n_share, 'ps', fin_currency, target_currency)
        
    if stock_name in filtered_df.index:
        stock_data = filtered_df.loc[stock_name]
        sector_name = stock_data['sector']
        industry_name = stock_data['industry']
        sector_df = filtered_df[filtered_df['sector'] == sector_name]
        sector_pe = (sector_df['mktCap'] * sector_df[pratio]).sum() / sector_df['mktCap'].sum()
        industry_df = filtered_df[filtered_df['industry'] == industry_name]
        industry_pe = (industry_df['mktCap'] * industry_df[pratio]).sum() / industry_df['mktCap'].sum()
    else:
        # For stocks not in table, we can't easily calculate sector average from filtered_df 
        # unless it belongs to one of the sectors in filtered_df
        stock_data = calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share)
        sector_name = stock_data['sector']
        industry_name = stock_data['industry']
        
        sector_df = filtered_df[filtered_df['sector'] == sector_name]
        if not sector_df.empty:
            sector_pe = (sector_df['mktCap'] * sector_df[pratio]).sum() / sector_df['mktCap'].sum()
        else:
            sector_pe = -1
            
        industry_df = filtered_df[filtered_df['industry'] == industry_name]
        if not industry_df.empty:
            industry_pe = (industry_df['mktCap'] * industry_df[pratio]).sum() / industry_df['mktCap'].sum()
        else:
            industry_pe = -1

    pe_ttm = pe_df['pe'].values[-1]
    current_price = price_df['close'].values[0]
    median_pe = pe_df['pe'].median()
    
    pe_dist_chart = hp.plot_pe_distribution(pe_df, pe_ttm, axis_label=ratio)
    val_cols[0].altair_chart(pe_dist_chart, width='stretch')
    pe_ts_chart = hp.plot_pe_timeseries(pe_df, axis_label=ratio)
    val_cols[1].altair_chart(pe_ts_chart, width='stretch')

    highlight_color = 'green' if pe_ttm <= median_pe else 'red'
    sector_color = 'green' if pe_ttm <= sector_pe else 'red'
    industry_color = 'green' if pe_ttm <= industry_pe else 'red'

    with val_cols[2]:
        ci = pe_df['pe'].quantile([.05, .95]).values
        markdown_table = f"""
        | Metric | Value |
        | ------ | ----- |
        | Current {ratio} | **:{highlight_color}[{pe_ttm:,.2f}]** |
        | Current Price | **:{highlight_color}[{int(current_price):,}]** |
        | Median last {year} year {ratio} | {median_pe:,.2f} |
        | Fair Price | {int((median_pe/pe_ttm)*current_price):,} |
        | 95% Confidence Interval range {ratio} | {ci[0]:,.2f} - {ci[1]:,.2f} |
        | 95% Confidence Interval range Price | {int((ci[0]/pe_ttm)*current_price):,} - {int((ci[1]/pe_ttm)*current_price):,} |
        """
        if industry_pe != -1 and sector_pe != -1:
            markdown_table += f"| Industry: {industry_name} {ratio} | **:{industry_color}[{industry_pe:,.2f}]** | \n \
        | Sector: {sector_name} {ratio} | **:{sector_color}[{sector_pe:,.2f}]** |"
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

def render_ddm_valuation(sdf, stock_name, filtered_df, fin=None, cp_df=None, price_df=None, n_share=None):
    if stock_name in filtered_df.index:
        stock_data = filtered_df.loc[stock_name]
    else:
        stock_data = calculate_missing_stats(stock_name, fin, cp_df, price_df, sdf, n_share)
        
    last_div = stock_data['lastDiv']
    current_price = stock_data['price']
    
    if sdf is not None and not sdf.empty:
        stats = hd.calc_div_stats(hd.preprocess_div(sdf))
        cagr_5y = stats.get('cagr_5y', None)
    else:
        cagr_5y = None
    
    default_g = min(cagr_5y, 5.0) if (cagr_5y is not None and cagr_5y > 0) else 2.5
    
    beta = stock_data.get('beta', 1.0)
    
    # Cost of Equity calculation using CAPM
    st.write("#### Cost of Equity (CAPM)")
    capm_cols = st.columns(3)
    rf_pct = capm_cols[0].number_input('Risk-Free Rate (%)', value=6.5 if stock_name.endswith('.JK') else 4.5, step=0.1, key=f"ddm_rf_{stock_name}", help="Yield of 10-year government bond")
    erp_pct = capm_cols[1].number_input('Equity Risk Premium (%)', value=5.0, step=0.1, key=f"ddm_erp_{stock_name}", help="Extra return expected from stocks over risk-free rate")
    beta_val = capm_cols[2].number_input('Beta', value=float(beta), step=0.01, key=f"ddm_beta_{stock_name}", help="Stock volatility relative to the market")
    
    capm_ke = rf_pct + (beta_val * erp_pct)
    st.info(f"**Calculated Cost of Equity ($K_e$)**: {rf_pct}% + ({beta_val:.2f} × {erp_pct}%) = **{capm_ke:.2f}%**")

    cols = st.columns(2)
    r_pct = cols[0].number_input(label='Required Rate of Return (Cost of Equity) %', value=float(capm_ke), min_value=1.0, max_value=50.0, step=0.5, key=f"ddm_r_{stock_name}")
    g_pct = cols[1].number_input(label='Terminal Dividend Growth Rate %', value=float(default_g), min_value=0.0, max_value=20.0, step=0.5, key=f"ddm_g_{stock_name}")
    
    if g_pct >= r_pct:
        st.error('Gordon Growth Model requires the Growth Rate to be strictly less than the Required Rate of Return.')
        return
        
    r = r_pct / 100.0
    g = g_pct / 100.0
    
    next_div = last_div * (1 + g)
    intrinsic_value = next_div / (r - g) if (r - g) > 0 else 0
    
    if current_price > 0:
        margin_of_safety = (intrinsic_value - current_price) / current_price
    else:
        margin_of_safety = 0
        
    diff = intrinsic_value / current_price if current_price > 0 else 1
    
    if intrinsic_value > current_price * 1.05:
        assessment = f'**:green[Undervalued]**. Margin of Safety: **:green[{margin_of_safety*100:.2f}%]**'
        highlight_color = 'green'
    elif intrinsic_value < current_price * 0.95:
        assessment = f'**:red[Overvalued]**. Premium: **:red[{abs(margin_of_safety)*100:.2f}%]**'
        highlight_color = 'red'
    else:
        assessment = '**Fair Valued**'
        highlight_color = 'green'
        
    ci_lower = intrinsic_value * 0.9
    ci_upper = intrinsic_value * 1.1

    markdown_table = f"""
    | Metric | Value |
    | ------ | ----- |
    | Current Price | **:{highlight_color}[{int(current_price):,}]** |
    | Next Expected Dividend (D1) | {next_div:,.2f} |
    | Intrinsic Value (DDM) | {int(intrinsic_value):,} |
    | Range Estimate | {int(ci_lower):,} - {int(ci_upper):,} |
    """
    
    res_cols = st.columns(2)
    with res_cols[0]:
        st.markdown(markdown_table)
        st.write(f'Assessment: {assessment}')

    # Heatmap generation
    data = []
    g_range = [max(0.0, g_pct + i) for i in [-2, -1, 0, 1, 2]]
    r_range = [max(1.0, r_pct + i) for i in [-2, -1, 0, 1, 2]]
    
    for r_val in r_range:
        for g_val in g_range:
            r_dec = r_val / 100.0
            g_dec = g_val / 100.0
            if r_dec > g_dec:
                val = (last_div * (1 + g_dec)) / (r_dec - g_dec)
            else:
                val = None
                
            if val is not None:
                data.append({
                    'Cost of Equity (r)': f"{r_val:.2f}%",
                    'Growth Rate (g)': f"{g_val:.2f}%",
                    'Intrinsic Value': val,
                    'r_val': r_val,
                    'g_val': g_val
                })
                
    if data:
        heatmap_df = pd.DataFrame(data)
        
        base = alt.Chart(heatmap_df).encode(
            x=alt.X('Growth Rate (g):O', sort=alt.EncodingSortField(field='g_val', order='ascending')),
            y=alt.Y('Cost of Equity (r):O', sort=alt.EncodingSortField(field='r_val', order='descending')),
        )
        
        heatmap = base.mark_rect().encode(
            color=alt.Color('Intrinsic Value:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=current_price), legend=None),
            tooltip=['Cost of Equity (r)', 'Growth Rate (g)', alt.Tooltip('Intrinsic Value:Q', format=',.0f')]
        )
        
        text = base.mark_text(baseline='middle').encode(
            text=alt.Text('Intrinsic Value:Q', format=',.0f'),
            color=alt.condition(
                alt.datum['Intrinsic Value'] > current_price,
                alt.value('black'),
                alt.value('white')
            )
        )
        
        chart = (heatmap + text).properties(
            title='Sensitivity: Intrinsic Value',
            height=300
        )
        
        with res_cols[1]:
            st.altair_chart(chart, width='stretch')

@st.cache_data(show_spinner=False)
def _calc_best_buy_cached(price_json, sdf_json):
    """Cached wrapper for calc_best_buy_timing — DataFrames passed as JSON."""
    price_df = pd.read_json(io.StringIO(price_json))
    sdf = pd.read_json(io.StringIO(sdf_json)) if sdf_json else None
    return hd.calc_best_buy_timing(price_df, sdf)


def render_best_buy_timing(price_df, sdf, stock_name):
    """
    Render two charts that answer "When is the best time to buy?":
      1. Monthly price seasonality (bar + IQR error band)
      2. Average price trajectory relative to ex-dividend date
    """
    import calendar as _cal

    try:
        sdf_json = sdf.to_json() if sdf is not None and not sdf.empty else None
        seasonality_df, pre_ex_df = _calc_best_buy_cached(price_df.to_json(), sdf_json)
    except Exception as e:
        st.warning(f'Could not compute buy-timing analysis: {e}')
        return

    if seasonality_df is None and pre_ex_df is None:
        st.info('Not enough historical data to compute buy-timing analysis (need ≥ 2 years of price data).')
        return

    chart_cols = st.columns(2)

    # ------------------------------------------------------------------ #
    # Chart 1 — Monthly Seasonality                                        #
    # ------------------------------------------------------------------ #
    with chart_cols[0]:
        st.markdown('#### 📅 Monthly Price Seasonality')
        st.caption('Relative price vs. annual average (< 100 = cheaper than average)')

        if seasonality_df is not None:
            month_order = list(_cal.month_abbr[1:])
            best_month_row = seasonality_df.loc[seasonality_df['mean'].idxmin()]
            best_month = best_month_row['month_name']
            best_val = best_month_row['mean']

            base = alt.Chart(seasonality_df)

            # IQR band
            band = base.mark_area(opacity=0.2, color='#2ecc71').encode(
                x=alt.X('month_name:O', sort=month_order, title='Month'),
                y=alt.Y('q25:Q', title='Relative Price (%)'),
                y2=alt.Y2('q75:Q'),
            )

            # Median line
            line = base.mark_line(point=True, color='#2ecc71', strokeWidth=2).encode(
                x=alt.X('month_name:O', sort=month_order),
                y=alt.Y('median:Q', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('month_name:O', title='Month'),
                    alt.Tooltip('mean:Q', title='Avg Relative Price', format='.2f'),
                    alt.Tooltip('median:Q', title='Median', format='.2f'),
                    alt.Tooltip('q25:Q', title='Q25', format='.2f'),
                    alt.Tooltip('q75:Q', title='Q75', format='.2f'),
                ]
            )

            # Reference line at 100
            ref = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
                color='#aaaaaa', strokeDash=[6, 4], strokeWidth=1
            ).encode(y='y:Q')

            # Highlight bar for cheapest month
            best_data = seasonality_df[seasonality_df['month_name'] == best_month]
            best_bar = alt.Chart(best_data).mark_bar(
                color='#27ae60', opacity=0.35, width=30
            ).encode(
                x=alt.X('month_name:O', sort=month_order),
                y=alt.Y('q25:Q'),
                y2=alt.Y2('q75:Q'),
            )

            chart = (band + best_bar + line + ref).properties(height=280)
            st.altair_chart(chart, width='stretch')

            st.success(f'🏆 **Best month to buy**: **{best_month}** — avg {best_val:.1f}% of annual price')
        else:
            st.info('Insufficient price history for monthly seasonality (need ≥ 2 years).')

    # ------------------------------------------------------------------ #
    # Chart 2 — Pre-Ex-Date Trajectory                                     #
    # ------------------------------------------------------------------ #
    with chart_cols[1]:
        st.markdown('#### 📉 Price Trajectory Around Ex-Dividend Date')
        st.caption('Normalised to ex-date = 100. Values < 100 before day 0 indicate a pre-ex dip.')

        if pre_ex_df is not None and not pre_ex_df.empty:
            # Clip to only days with meaningful coverage (avoid sparse edges)
            pre_ex_plot = pre_ex_df[
                (pre_ex_df['days_to_ex'] >= -180) & (pre_ex_df['days_to_ex'] <= 30)
            ].copy()

            base = alt.Chart(pre_ex_plot)

            band = base.mark_area(opacity=0.2, color='#3498db').encode(
                x=alt.X('days_to_ex:Q', title='Days to Ex-Date'),
                y=alt.Y('q25:Q', title='Relative Price (ex-date = 100)'),
                y2=alt.Y2('q75:Q'),
            )

            line = base.mark_line(color='#3498db', strokeWidth=2).encode(
                x=alt.X('days_to_ex:Q'),
                y=alt.Y('mean:Q', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('days_to_ex:Q', title='Days to Ex-Date'),
                    alt.Tooltip('mean:Q', title='Avg Relative Price', format='.2f'),
                    alt.Tooltip('median:Q', title='Median', format='.2f'),
                ]
            )

            # Ex-date vertical rule
            ex_rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                color='#e74c3c', strokeDash=[5, 4], strokeWidth=2
            ).encode(x='x:Q')

            ex_label = alt.Chart(pd.DataFrame({'x': [2], 'y': [pre_ex_plot['mean'].max()], 'text': ['Ex-Date']})).mark_text(
                align='left', color='#e74c3c', fontSize=11
            ).encode(x='x:Q', y='y:Q', text='text:N')

            ref = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
                color='#aaaaaa', strokeDash=[6, 4], strokeWidth=1
            ).encode(y='y:Q')

            chart = (band + line + ref + ex_rule + ex_label).properties(height=280)
            st.altair_chart(chart, width='stretch')

            # Find the best dip point
            pre_only = pre_ex_plot[pre_ex_plot['days_to_ex'] < 0]
            if not pre_only.empty:
                best_dip = pre_only.loc[pre_only['mean'].idxmin()]
                dip_day = int(best_dip['days_to_ex'])
                dip_val = best_dip['mean']
                st.success(
                    f'🎯 **Best buy window**: ~**{abs(dip_day)} days before** ex-date '
                    f'(avg {100 - dip_val:.1f}% cheaper than ex-date price)'
                )
        else:
            st.info('No dividend history available to compute ex-date trajectory.')


def render_compounding_simulation(stock_name, price_df, sdf):

    this_year = datetime.now().year
    cols = st.columns(4)
    start_year = cols[0].number_input(label='Start Year', value=2021, min_value=2010, max_value=this_year-2, key=f"sim_start_{stock_name}")
    end_year = cols[1].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1, key=f"sim_end_{stock_name}")
    initial_value = cols[2].number_input(label='Initial investment (million)', value=10, min_value=1, max_value=1000, key=f"sim_init_{stock_name}")
    monthly_topup = cols[3].number_input(label='Monthly Top-up (million)', value=1, min_value=0, max_value=100, key=f"sim_monthly_{stock_name}")

    porto, activities = hd.simulate_dividend_compounding(
        stock_name, price_df, sdf,
        start_year, end_year,
        initial_value * 1_000_000,
        monthly_topup * 1_000_000,
    )

    if not porto:
        st.warning('No simulation data available for the selected period.')
        return

    # ── Build portfolio timeline DataFrame ─────────────────────────────── #
    porto_df = pd.DataFrame(porto)
    porto_df['date']      = pd.to_datetime(porto_df['date'])
    porto_df['cost']      = porto_df['num_stock'] * porto_df['price'] * 100
    porto_df['cum_stock'] = porto_df['num_stock'].cumsum()
    porto_df['cum_cost']  = porto_df['cost'].cumsum()
    porto_df['mkt_value'] = porto_df['price'] * porto_df['cum_stock'] * 100

    # ── Parse dividend income from activity log ─────────────────────────── #
    div_rows = []
    for act in activities:
        if 'receive dividend' in act:
            try:
                parts = act.split()
                act_date = pd.to_datetime(parts[0])
                total_idx = parts.index('Total') + 1
                total_div = float(parts[total_idx])
                div_rows.append({'date': act_date, 'dividend_income': total_div})
            except Exception:
                pass
    div_income_df = pd.DataFrame(div_rows) if div_rows else pd.DataFrame(columns=['date', 'dividend_income'])

    # ── Key metrics ────────────────────────────────────────────────────── #
    n_years         = max(end_year - start_year, 1)
    total_invested  = initial_value * 1_000_000 + monthly_topup * 1_000_000 * n_years * 12
    final_mkt_value = porto_df['mkt_value'].iloc[-1]
    total_dividends = div_income_df['dividend_income'].sum() if not div_income_df.empty else 0
    final_lots      = int(porto_df['cum_stock'].iloc[-1])
    avg_price       = porto_df['price'].mean()
    div_lots_approx = int(total_dividends / (avg_price * 100)) if avg_price > 0 else 0
    cagr            = ((final_mkt_value / total_invested) ** (1 / n_years) - 1) * 100 if total_invested > 0 else 0
    total_return_pct = (final_mkt_value / total_invested - 1) * 100 if total_invested > 0 else 0

    # ── Hero KPI row ───────────────────────────────────────────────────── #
    st.markdown('---')
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('💰 Total Invested',          f'Rp {total_invested/1e6:,.1f}M')
    delta_color = 'normal' if final_mkt_value >= total_invested else 'inverse'
    k2.metric('📈 Portfolio Value',          f'Rp {final_mkt_value/1e6:,.1f}M',
              delta=f'{total_return_pct:+.1f}%', delta_color=delta_color)
    k3.metric('🎁 Total Dividends Received', f'Rp {total_dividends/1e6:,.1f}M')
    k4.metric('📊 CAGR',                     f'{cagr:.1f}%/yr')
    k5.metric('🎯 Bonus Lots from Dividends', f'~{div_lots_approx:,} lots',
              help='Approximate lots purchased using reinvested dividends')
    st.markdown('---')

    if div_lots_approx > 0:
        pct_from_div = div_lots_approx / final_lots * 100 if final_lots > 0 else 0
        st.success(
            f"🚀 **The Power of Compounding:** Over {n_years} years, reinvesting dividends bought you "
            f"approximately **{div_lots_approx:,} bonus lots** (~{pct_from_div:.0f}% of your total "
            f"{final_lots:,} lots), contributing **Rp {total_dividends/1e6:,.1f}M** to your portfolio "
            f"— *without any extra money from your pocket!*"
        )

    chart_col1, chart_col2 = st.columns([3, 2])

    # ── Chart 1: Stacked area — cost basis vs gains ─────────────────────── #
    with chart_col1:
        st.markdown('#### 📈 Portfolio Growth Over Time')
        st.caption('Blue = cash you invested. Green = total portfolio value (incl. dividend compounding gains).')

        cost_area = alt.Chart(porto_df).mark_area(
            opacity=0.55, color='#5b8dee', interpolate='monotone'
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('cum_cost:Q', title='Value (Rp)', stack=None),
            tooltip=[
                alt.Tooltip('date:T', title='Date'),
                alt.Tooltip('cum_cost:Q', title='Total Invested', format=',.0f'),
            ]
        )
        mkt_area = alt.Chart(porto_df).mark_area(
            opacity=0.40, color='#27ae60', interpolate='monotone'
        ).encode(
            x=alt.X('date:T'),
            y=alt.Y('mkt_value:Q', stack=None),
        )
        mkt_line = alt.Chart(porto_df).mark_line(
            color='#1e8449', strokeWidth=2.5, interpolate='monotone'
        ).encode(
            x=alt.X('date:T'),
            y=alt.Y('mkt_value:Q'),
            tooltip=[
                alt.Tooltip('date:T', title='Date'),
                alt.Tooltip('mkt_value:Q', title='Portfolio Value', format=',.0f'),
                alt.Tooltip('cum_cost:Q', title='Total Invested', format=',.0f'),
            ]
        )
        cost_line = alt.Chart(porto_df).mark_line(
            color='#2471a3', strokeWidth=1.5, strokeDash=[5, 3], interpolate='monotone'
        ).encode(
            x=alt.X('date:T'),
            y=alt.Y('cum_cost:Q'),
        )
        st.altair_chart((cost_area + mkt_area + mkt_line + cost_line).properties(height=300), width='stretch')

    # ── Chart 2: Annual dividend income snowball ─────────────────────────── #
    with chart_col2:
        st.markdown('#### 🎁 Annual Dividend Income')
        st.caption('Growing each year as you accumulate more shares — the snowball effect!')

        if not div_income_df.empty:
            div_income_df['year'] = div_income_df['date'].dt.year
            annual_div = div_income_df.groupby('year')['dividend_income'].sum().reset_index()
            annual_div['year_str'] = annual_div['year'].astype(str)

            bar = alt.Chart(annual_div).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#f39c12'
            ).encode(
                x=alt.X('year_str:O', title='Year', sort=None),
                y=alt.Y('dividend_income:Q', title='Dividend Income (Rp)'),
                tooltip=[
                    alt.Tooltip('year_str:O', title='Year'),
                    alt.Tooltip('dividend_income:Q', title='Annual Dividend', format=',.0f'),
                ]
            )
            trend = alt.Chart(annual_div).mark_line(
                color='#e67e22', strokeWidth=2, strokeDash=[4, 2]
            ).encode(
                x=alt.X('year_str:O', sort=None),
                y=alt.Y('dividend_income:Q'),
            )
            st.altair_chart((bar + trend).properties(height=280), width='stretch')
        else:
            st.info('No dividend events recorded in this period.')

    # ── Chart 3: Strategy Comparison ────────────────────────────────────── #
    st.markdown('#### ⚖️ Strategy Comparison: Reinvest vs. Pocket Dividends vs. Buy-and-Hold Only')
    st.caption('What if you had not reinvested dividends, or had bought a non-dividend stock instead?')

    start_price     = porto_df['price'].iloc[0]
    start_lots      = int((initial_value * 1_000_000) / (start_price * 100)) if start_price > 0 else 0
    porto_df['hold_only_value'] = start_lots * porto_df['price'] * 100

    compare_df = pd.DataFrame({
        'date':                          porto_df['date'],
        'DRIP (Reinvest Dividends)':     porto_df['mkt_value'],
        'Cash (Pocket Dividends)':       porto_df['cum_cost'],
        'Hold Only (No Dividend Stock)': porto_df['hold_only_value'],
    }).melt(id_vars='date', var_name='Strategy', value_name='Value')

    compare_chart = alt.Chart(compare_df).mark_line(strokeWidth=2.5, interpolate='monotone').encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Value:Q', title='Portfolio Value (Rp)', scale=alt.Scale(zero=False)),
        color=alt.Color('Strategy:N', scale=alt.Scale(
            domain=['DRIP (Reinvest Dividends)', 'Cash (Pocket Dividends)', 'Hold Only (No Dividend Stock)'],
            range=['#27ae60', '#2980b9', '#95a5a6']
        )),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('Strategy:N'),
            alt.Tooltip('Value:Q', format=',.0f'),
        ]
    ).properties(height=280)
    st.altair_chart(compare_chart, width='stretch')

    final_drip  = porto_df['mkt_value'].iloc[-1]
    final_cash  = porto_df['cum_cost'].iloc[-1]
    final_hold  = porto_df['hold_only_value'].iloc[-1]
    st.markdown(f"""
| Strategy | Final Value | Gap vs DRIP |
|---|---|---|
| 🟢 DRIP (Reinvest Dividends) | **Rp {final_drip/1e6:,.1f}M** | — |
| 🔵 Cash (Pocket Dividends) | Rp {final_cash/1e6:,.1f}M | **Rp {(final_drip-final_cash)/1e6:,.1f}M less** |
| ⚪ Hold Only (No Dividend Stock) | Rp {final_hold/1e6:,.1f}M | **Rp {(final_drip-final_hold)/1e6:,.1f}M less** |
    """)

    with st.expander('📋 Raw Transaction Log', expanded=False):
        st.dataframe(pd.DataFrame({'Activity': activities}), hide_index=True)

def render_classic_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, sl):
    with st.expander('Company Profile', expanded=False):    
        render_company_profile(cp_df, stock_name)
    
    currency = cp_df.loc[stock_name, 'currency']

    with st.expander(f'Dividend History: {stock_name}', expanded=True):
        render_dividend_history(sdf, filtered_df, stock_name, filtered_df, fin=fin, n_share=n_share, currency=currency, cp_df=cp_df, price_df=price_df)

    with st.expander(f'Best Time to Buy: {stock_name}', expanded=True):
        render_best_buy_timing(price_df, sdf, stock_name)
        
    with st.expander(f'Financial Information: {stock_name}', expanded=True):
        render_financial_info(fin, currency, stock_name, filtered_df)
        
    with st.expander('Price Movement', expanded=True):
        render_price_movement(price_df)
        
    with st.expander(f'Valuation Analysis: {stock_name}', expanded=True):
        render_valuation_analysis(price_df, fin, n_share, sl, stock_name, filtered_df, cp_df=cp_df, sdf=sdf)

    with st.expander(f'Dividend Discount Model Valuation: {stock_name}', expanded=True):
        render_ddm_valuation(sdf, stock_name, filtered_df, fin=fin, cp_df=cp_df, price_df=price_df, n_share=n_share)

    with st.expander(f'Compounding Simulation: {stock_name}', expanded=True):
        render_compounding_simulation(stock_name, price_df, sdf)




default_sl = 0
if 'market' in st.query_params:
    market_param = st.query_params['market']
    if market_param in ['S&P500', 'SP500', 'US']:
        default_sl = 1


# sl = st.sidebar.radio('Stock List', ['JKSE', 'S&P500'], horizontal=True, key='sl', index=default_sl)

stock_select = st.sidebar.radio(
    'Stock List Selection',
    ['Indonesian Stock', 'S&P 500 (US and World Stock)'],
    horizontal=False,
    key='sl',
    index=default_sl
)

if stock_select == 'Indonesian Stock':
    sl = 'JKSE'
else:
    sl = 'S&P500'

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


logger = get_logger('screener')

final_df = get_div_score_table(key)
if sl != 'JKSE':
    final_df = final_df.drop('GOOGL')

if sl == 'JKSE':
    is_syariah = st.sidebar.toggle('Syariah Only?')
    if is_syariah:
        final_df = final_df[final_df['is_syariah'] == True]

# ---------------------------------------------------------------------------
# Column pruning — keep only columns actually used in the page
# ---------------------------------------------------------------------------
_KEEP_COLS = [
    # Identifiers / display
    'price', 'changes', 'sector', 'industry', 'mktCap', 'ipoDate',
    # Dividend
    'yield', 'lastDiv', 'avgFlatAnnualDivIncrease', 'numDividendYear',
    'positiveYear', 'numOfYear', 'maximumCutPct', 'max10CutPct',
    # Financial
    'peRatio', 'psRatio', 'revenueGrowth', 'netIncomeGrowth',
    'medianProfitMargin', 'earningTTM', 'revenueTTM',
    'revenueGrowthTTM', 'netIncomeGrowthTTM', 'beta',
    # Returns
    'return_7d', 'return_1m', 'return_1y', 'return_10y',
    'total_return_1y', 'total_return_10y',
    # Syariah flag (JKSE only)
    'is_syariah',
]
_keep = [c for c in _KEEP_COLS if c in final_df.columns]
final_df = final_df[_keep]

# Show total app RAM in the sidebar
_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
st.sidebar.caption(f'RAM: {_rss_mb:.0f} MB')


# ---------------------------------------------------------------------------
# Cached compute pipeline — only re-runs when final_df changes
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_df(df):
    """Compute all derived columns and sort. Cached so it only runs when data changes."""
    df = df.copy()

    df['marginTTM'] = df['earningTTM'] / df['revenueTTM'] * 100
    df['mc_penalty'] = df['mktCap'].apply(lambda x: 1 / (1 + np.exp(-2 * (x / 3_000_000_000_000 - 1))))

    df['maximumCutPct'] = df['maximumCutPct'].apply(lambda x: min(x, 0) * -1)
    df['max10CutPct']   = df['max10CutPct'].apply(lambda x: min(x, 0) * -1)
    df['maxDivIncrease']       = df.apply(lambda x: min(x['avgFlatAnnualDivIncrease'], x['lastDiv'] * 0.05), axis=1)
    df['maxRevGrowthDecrease'] = df.apply(lambda x: min(x['revenueGrowthTTM'], 0), axis=1)
    df['maxIncGrowthDecrease'] = df.apply(lambda x: min(x['netIncomeGrowthTTM'], 0), axis=1)

    return_cols = ['return_7d', 'return_1m', 'return_1y', 'return_10y', 'total_return_1y', 'total_return_10y']
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col] * 100

    df['DScore'] = (
        (df['lastDiv'] + df['maxDivIncrease'] * PROJECTION_HORIZON_YRS * (df['positiveYear'] / df['numOfYear'])) / df['price']
    ) * 100 \
      * (df['numDividendYear'] / (df['numDividendYear'] + DIV_MATURITY_HALFLIFE)) \
      * (1 - np.exp(-df['numDividendYear'] / 5)) \
      * (100 - df['max10CutPct']) / 100 \
      * df['mc_penalty'] \
      * (1 + df['maxRevGrowthDecrease'] / 100) \
      * (1 + df['maxIncGrowthDecrease'] / 100)

    df = df.fillna(0).sort_values('DScore', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


default_view = 'Table'
color_var_index = 1
if 'view' in st.query_params:
    if st.query_params['view'] == 'treemap':
        default_view = 'Treemap'
        if 'color_var' in st.query_params:
            if st.query_params['color_var'] == 'return':
                color_var_index = 5

full_table_section = st.container(border=True)
with full_table_section:

    filtered_df = get_processed_df(final_df)

    if sl == 'JKSE':
        divisor = 1_000_000_000_000
        mcap_label = 'Market Cap (T IDR)'
        mcap_format = '%,.2f T'
        earning_label = 'Earning TTM (T IDR)'
        revenue_label = 'Revenue TTM (T IDR)'
        fin_format = '%,.2f T'
    else:
        divisor = 1_000_000_000
        mcap_label = 'Market Cap (B USD)'
        mcap_format = '%,.2f B'
        earning_label = 'Earning TTM (B USD)'
        revenue_label = 'Revenue TTM (B USD)'
        fin_format = '%,.2f B'

    # Build display-only columns without mutating the cached filtered_df
    display_df = filtered_df.copy()
    mcap_pos    = display_df.columns.get_loc('mktCap')
    earning_pos = display_df.columns.get_loc('earningTTM')
    revenue_pos = display_df.columns.get_loc('revenueTTM')
    display_df.insert(mcap_pos,         'mktCapDisplay',    display_df['mktCap']     / divisor)
    display_df.insert(earning_pos + 1,  'earningTTMDisplay', display_df['earningTTM'] / divisor)
    display_df.insert(revenue_pos + 2,  'revenueTTMDisplay', display_df['revenueTTM'] / divisor)

    # Measure the fully processed (cached) filtered_df
    # (removed — measurement moved to sidebar as process RSS)

    top_cols = st.columns(2)
    view = top_cols[0].segmented_control(label='View Option', 
                         options=['Table', 'Treemap', 'Scatter Plot', 'Distribution'],
                         selection_mode='single',
                         default=default_view)


    if view == 'Table':

        cfig={  # noqa: E225
            'Rank': st.column_config.NumberColumn(
                'Rank',
                help='Rank based on Dividend Score',
                format='%d',
            ),
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
                format='%,.0f',
            ),
            'beta': st.column_config.NumberColumn(
                'Beta',
                help='Stock Beta (Risk relative to market)',
                format='%.2f',
            ),
            'yield': st.column_config.NumberColumn(
                'Dividend Yield',
                help='The dividend yield on the current price (in pct)',
                min_value=0,
                max_value=100,
                step=0.01,
                format='%,.02f',
            ),
            'sector': st.column_config.TextColumn(
                'Sector',
                help='The sector of the stock',
            ),
            'industry': st.column_config.TextColumn(
                'Industry',
                help='The industry of the stock',
            ),
            'mktCap': None,
            'mktCapDisplay': st.column_config.NumberColumn(
                mcap_label,
                help='Market Capitalization on the current price',
                format=mcap_format,
            ),
            'ipoDate': st.column_config.DateColumn(
                'IPO Date',
                help='IPO Date',
            ),
            'revenueGrowth': st.column_config.NumberColumn(
                'Revenue Growth',
                help='Average revenue growth in the last 5 years',
                format='%,.02f',
            ),
            'revenueGrowthTTM': st.column_config.NumberColumn(
                'Revenue Growth (TTM)',
                help='Revenue growth in the last twelve months',
                format='%,.02f',
            ),
            'netIncomeGrowth': st.column_config.NumberColumn(
                'Income Growth',
                help='Average net income growth in the last 5 years',
                format='%,.02f',
            ),
            'netIncomeGrowthTTM': st.column_config.NumberColumn(
                'Income Growth (TTM)',
                help='Net income growth in the last twelve months',
                format='%,.02f',
            ),
            'medianProfitMargin': st.column_config.NumberColumn(
                'Profit Margin',
                help='Median profit margin in the last 5 years',
                format='%,.02f',
            ),
            'avgFlatAnnualDivIncrease': st.column_config.NumberColumn(
                'Dividend Growth',
                help='Average annual dividend increase in the last 5 years',
                format='%,.02f',
            ),
            'numOfYear': st.column_config.NumberColumn(
                'Num of Year',
                help='Number of years the stock has been listed',
                format='%,d',
            ),
            'numDividendYear': st.column_config.NumberColumn(
                'Num of Dividend Year',
                help='Number of years the stock has paid dividend',
                format='%,d',
            ),
            'positiveYear': st.column_config.NumberColumn(
                'Positive Year',
                help='Number of years the stock has increased dividend',
                format='%,d',
            ),
            'score': st.column_config.NumberColumn(
                'Score',
                help='Score of the stock',
                format='%,.02f',
            ),
            'DScore': st.column_config.NumberColumn(
                'Dividend Score',
                help='Dividend Score of the stock',
                format='%,.02f',
            ),
            'lastDiv': st.column_config.NumberColumn(
                'Last Dividend',
                help='Last Dividend Paid in Last/Current Fiscal Year',
                format='%,.02f',
            ),
            'earningTTM': None,
            'earningTTMDisplay': st.column_config.NumberColumn(
                earning_label,
                help='Earning in the last twelve months',
                format=fin_format,
            ),
            'revenueTTM': None,
            'revenueTTMDisplay': st.column_config.NumberColumn(
                revenue_label,
                help='Revenue in the last twelve months',
                format=fin_format,
            ),
            'peRatio': st.column_config.NumberColumn(
                'PE Ratio',
                help='Price to Earnings Ratio',
                format='%,.2f'
            ),
            'psRatio': st.column_config.NumberColumn(
                'PS Ratio',
                help='Price to Sales/Revenue Ratio',
                format='%.2f'
            ),
            'return_7d': st.column_config.NumberColumn(
                '7D Price Return',
                help='Price return in the last week',
                format='%.2f%%',
            ),
            'return_1m': st.column_config.NumberColumn(
                '1M Price Return',
                help='Price return in the last month',
                format='%.2f%%',
            ),
            'return_1y': st.column_config.NumberColumn(
                '1Y Price Return',
                help='Price return in the last 1 year',
                format='%.2f%%',
            ),
            'return_10y': st.column_config.NumberColumn(
                '10Y Price Return',
                help='Price return in the last 10 years',
                format='%.2f%%',
            ),
            'total_return_1y': st.column_config.NumberColumn(
                'Total 1Y Return',
                help='Total return (Price + Dividend) in the last 1 year',
                format='%.2f%%',
            ),
            'total_return_10y': st.column_config.NumberColumn(
                'Total 10Y Return',
                help='Total return (Price + Dividend) in the last 10 years',
                format='%.2f%%',
            )

        }

        event = st.dataframe(display_df, selection_mode=['single-row'], on_select='rerun', column_config=cfig)

    elif view == 'Treemap':
        
        treemap_cols = st.columns([2,2,3,2])
        size_var = treemap_cols[0].selectbox(options=['Market Cap', 'Revenue', 'Net Income', 'Dividend Yield'], label='Select Size Variable')
        color_var = treemap_cols[1].selectbox(options=['None', 'Dividend Yield', 'Median Profit Margin', 'TTM Profit Margin', 'Revenue Growth', '1D Price Return', '7D Price Return', '1M Price Return', '1Y Price Return', '10Y Price Return', 'Total 1Y Return', 'Total 10Y Return', 'PE Ratio', 'PS Ratio'], label='Select Color Variable', index=color_var_index)
        sector_var = treemap_cols[2].selectbox(options=['ALL']+filtered_df['sector'].unique().tolist(), label='Select Sector')
        group_secs = treemap_cols[3].toggle('Group by Sector', value=True)
        
        # Build df_tree in one shot — avoids 17 separate column-assignment intermediate arrays
        df_tree = pd.DataFrame({
            'sector':              filtered_df['sector'],
            'industry':            filtered_df['industry'],
            'Market Cap':          filtered_df['mktCap'] / 1_000_000_000,
            'Revenue':             filtered_df['revenueTTM'],
            'Net Income':          filtered_df['earningTTM'],
            'Dividend Yield':      filtered_df['yield'],
            'Median Profit Margin': filtered_df['medianProfitMargin'],
            'TTM Profit Margin':   filtered_df['marginTTM'],
            'Revenue Growth':      filtered_df['revenueGrowth'],
            '1D Price Return':     filtered_df['changes'] / filtered_df['price'] * 100,
            '7D Price Return':     filtered_df['return_7d'],
            '1M Price Return':     filtered_df['return_1m'],
            '1Y Price Return':     filtered_df['return_1y'],
            '10Y Price Return':    filtered_df['return_10y'],
            'Total 1Y Return':     filtered_df['total_return_1y'],
            'Total 10Y Return':    filtered_df['total_return_10y'],
            'PE Ratio':            filtered_df['peRatio'],
            'PS Ratio':            filtered_df['psRatio'],
        }, index=filtered_df.index).dropna()

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
        if color_var in ['1D Price Return', '7D Price Return', '1M Price Return', '1Y Price Return', '10Y Price Return', 'Total 1Y Return', 'Total 10Y Return']:
            color_map = 'red_green'
            color_threshold = [-3, -1, 0, 1, 3]
        elif color_var == 'PE Ratio':
            color_map = 'red_shade'
            color_threshold = [-100, 0, 5, 15]
        elif color_var == 'PS Ratio':
            color_map = 'red_shade'
            color_threshold = [-1000, -100, -10, -1, 0, 1, 2, 3, 5]
        elif color_var == 'Dividend Yield':
            color_threshold = [0,1,2,3,4, 5, 6, 7,8,9,10]

        tree_data = hd.prep_treemap(df_tree, size_var=size_var, color_var=color_var, color_threshold=color_threshold, add_label=add_label, group_secs=group_secs)
        option = hp.plot_treemap(tree_data, size_var=size_var, color_var=color_var, show_gradient=show_gradient, colormap=color_map, group_secs=group_secs)
        
        click_event_js = """function(params){console.log('Clicked item:',params.name);return params.name;}"""
        
        clicked_item_name = st_echarts(
            option,
            events={"click": click_event_js},
            height='900px', 
            width='100%'
        )
    

    elif view == 'Scatter Plot':

        sp_options = {
            'PE Ratio': 'peRatio',
            'PS Ratio': 'psRatio',
            'Dividend Yield': 'yield',
            'Market Cap': 'mktCap',
            'Revenue Growth': 'revenueGrowth',
            'Net Income Growth': 'netIncomeGrowth',
            'Profit Margin': 'medianProfitMargin',
            'Num Dividend Year': 'numDividendYear'
        }

        sp_cols = st.columns(3)
        x_metric = sp_cols[0].selectbox('X Axis', options=list(sp_options.keys()), index=0)
        y_metric = sp_cols[1].selectbox('Y Axis', options=list(sp_options.keys()), index=2)
        size_metric = sp_cols[2].selectbox('Size', options=list(sp_options.keys()), index=3)

        x_col = sp_options[x_metric]
        y_col = sp_options[y_metric]
        size_col = sp_options[size_metric]

        # Handle outliers for better visualization: remove top and bottom 5%
        # This ensures the distribution isn't squashed by extreme values
        q95_x = filtered_df[x_col].quantile(0.95)
        q05_x = filtered_df[x_col].quantile(0.05)
        q95_y = filtered_df[y_col].quantile(0.95)
        q05_y = filtered_df[y_col].quantile(0.05)

        plot_df = filtered_df[
            (filtered_df[x_col] <= q95_x) & (filtered_df[x_col] >= q05_x) &
            (filtered_df[y_col] <= q95_y) & (filtered_df[y_col] >= q05_y)
        ]

        sp = alt.Chart(plot_df.reset_index()).mark_circle().encode(
            x=alt.X(x_col, title=x_metric),
            y=alt.Y(y_col, title=y_metric),
            size=alt.Size(size_col, title=size_metric),
            color='sector',
            tooltip=[
                'stock', 
                alt.Tooltip(x_col, title=x_metric, format='.2f'), 
                alt.Tooltip(y_col, title=y_metric, format='.2f'),
                alt.Tooltip(size_col, title=size_metric, format='.2f')
            ]
        ).interactive()
        st.altair_chart(sp, width='stretch')

    elif view == 'Distribution':

        dist_options = {
            'PE Ratio': 'peRatio',
            'PS Ratio': 'psRatio',
            'Dividend Yield': 'yield',
            'Market Cap': 'mktCap',
            'Revenue Growth': 'revenueGrowth',
            'Net Income Growth': 'netIncomeGrowth',
            'Profit Margin': 'medianProfitMargin',
            'Num Dividend Year': 'numDividendYear'
        }
        
        col1, col2, col3 = st.columns([1, 2, 1])
        selected_dist = col1.selectbox('Select Metric', options=list(dist_options.keys()))
        exclude_zero = col1.toggle('Exclude 0% Yield Stocks', value=False)
        col_name = dist_options[selected_dist]

        if exclude_zero:
            filtered_df = filtered_df[filtered_df['yield'] > 0]

        # Add comparison stocks multiselect
        comparison_stocks = col2.multiselect('Compare with specific stocks', options=filtered_df.index.tolist())
        
        comparison_vals = {}
        if comparison_stocks:
            for s in comparison_stocks:
                val = filtered_df.loc[s, col_name]
                comparison_vals[s] = val
        
        color_map = {
            'PE Ratio': 'green',
            'PS Ratio': 'green',
            'Dividend Yield': 'green',
            'Revenue Growth': 'green',
            'Net Income Growth': 'green',
            'Profit Margin': 'green',
            'Num Dividend Year': 'green',
            'Market Cap': 'green'
        }
        
        # Add range number inputs in col3
        min_data = float(filtered_df[col_name].min())
        max_data = float(filtered_df[col_name].max())
        q05 = float(filtered_df[col_name].quantile(0.05))
        q95 = float(filtered_df[col_name].quantile(0.95))
        
        with col3:
            st.write("Zoom Range")
            z_col1, z_col2 = st.columns(2)
            z_min = z_col1.number_input("Min", value=q05, min_value=min_data, max_value=max_data, key=f"z_min_{col_name}")
            z_max = z_col2.number_input("Max", value=q95, min_value=min_data, max_value=max_data, key=f"z_max_{col_name}")
            x_range = (z_min, z_max)
            fill_opacity = st.slider("Fill Opacity", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key=f"fill_opacity_{col_name}",
                                     help="Set to 0 to disable the area fill (useful when many stock labels overlap the curve)")

        dist_chart = hp.plot_card_distribution(
            filtered_df, 
            col_name, 
            current_val=None, 
            color=color_map.get(selected_dist, 'green'), 
            height=400, 
            show_axis=True,
            comparison_vals=comparison_vals if comparison_vals else None,
            x_range=x_range,
            fill_opacity=fill_opacity
        )
        st.altair_chart(dist_chart, width="stretch")


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

if sl == 'JKSE':
    if stock_name:
        stock_name = stock_name.upper()
        if '.JK' not in stock_name:
            stock_name += '.JK'

    stock_options = sorted(final_df.index.tolist())
    try:
        default_idx = stock_options.index(stock_name) if stock_name else None
    except ValueError:
        default_idx = None

    select_stock = st.selectbox(
        label='Click the checkbox on the leftside of the table above or select a stock from the list to get detailed information',
        options=stock_options,
        index=default_idx,
        placeholder="Type or select a stock..."
    )

    if select_stock:
        stock_name = select_stock
    else:
        st.stop()
else:
    select_stock = st.text_input(
        label='Click the checkbox on the leftside of the table above or type the name of the stock to get detailed information',
        value=stock_name
    )

    if select_stock:
        stock_name = select_stock.upper()
    else:
        st.stop()

progress_bar = st.progress(0, text='Downloading stock data... Please wait')
try:
    fin, cp_df, price_df, sdf, n_share = get_specific_stock_detail(stock_name, sl)
except Exception as e:
    logger.error(f'Error in downloading data for {stock_name}: {e}')
    st.error(
        f'Cannot find the stock {stock_name}. Please check the name again and dont forget to add exchange code at the end. '
        'For example .JK for Indonesian stock, .SI for Singaporean stock, .T for Japanese Stock, etc.',
        icon='🚨'
    )
    progress_bar.empty()
    st.stop()
progress_bar.empty()


default_dashboard = False
if 'overview' in st.query_params and st.query_params['overview'].lower() == 'true':
    default_dashboard = True

stock_view_mode = st.toggle("Dashboard View", value=default_dashboard)
mode = "Dashboard" if stock_view_mode else "Classic"

if mode == "Classic":
    render_classic_view(stock_name, final_df, fin, cp_df, price_df, sdf, n_share, sl)
else:
    render_dashboard_view(stock_name, filtered_df, fin, cp_df, price_df, sdf, n_share, final_df)
