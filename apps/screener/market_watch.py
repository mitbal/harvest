import os
import io
import json
import time
import logging
from datetime import date, datetime, timedelta

import redis
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts5 import st_echarts
from supabase import create_client

import harvest.plot as hp
import harvest.data as hd
from harvest.utils import setup_logging


st.title('📡 Market Watch')
st.caption('Daily price return heatmap — browse any trading day to see who moved and by how much.')

api_key = os.environ['FMP_API_KEY']
redis_url = os.environ['REDIS_URL']


# ── Helpers ──────────────────────────────────────────────────────────────── #

@st.cache_resource
def get_logger(name, level=logging.INFO):
    return setup_logging(name, level)


@st.cache_resource
def connect_redis(url):
    return redis.from_url(
        url,
        socket_connect_timeout=10,
        socket_timeout=30,
        socket_keepalive=True,
        retry_on_timeout=True
    )


@st.cache_resource
def get_supabase():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        return None
    return create_client(url, key)


logger = get_logger('market_watch')


# ── Load stock universe (sector / industry / mcap metadata) ──────────────── #

@st.cache_data(ttl=60 * 10, show_spinner='Loading stock universe…')
def get_stock_universe(key: str) -> pd.DataFrame:
    r = connect_redis(redis_url)
    rjson = r.get(key)
    if rjson is None:
        return pd.DataFrame()

    if isinstance(rjson, bytes) and rjson.startswith(b'PAR1'):
        df = pd.read_parquet(io.BytesIO(rjson))
    else:
        raw = json.loads(rjson)
        if isinstance(raw, dict) and 'content' in raw:
            df = pd.DataFrame(json.loads(raw['content']))
        else:
            df = pd.DataFrame(raw)

    df.rename(columns={'symbol': 'stock'}, inplace=True)
    return df.set_index('stock')


# ── Load historical prices for a date range from Supabase ────────────────── #

@st.cache_data(ttl=60 * 60, show_spinner='Fetching historical prices…')
def get_prices_for_date_range(
    symbols: tuple,       # tuple so it is hashable for caching
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """
    Fetches close prices for ``symbols`` between ``date_from`` and ``date_to``
    from the Supabase ``historical_prices`` table.
    Returns a DataFrame with columns [symbol, date, close].
    """
    supabase = get_supabase()
    if supabase is None:
        return pd.DataFrame(columns=['symbol', 'date', 'close'])

    all_rows = []
    sym_list = list(symbols)
    chunk_size = 50          # Supabase IN clause limit is generous, keep chunks small
    limit = 10_000

    for i in range(0, len(sym_list), chunk_size):
        chunk = sym_list[i: i + chunk_size]
        offset = 0
        while True:
            try:
                resp = (
                    supabase.table('historical_prices')
                    .select('symbol,date,close')
                    .in_('symbol', chunk)
                    .gte('date', date_from)
                    .lte('date', date_to)
                    .range(offset, offset + limit - 1)
                    .execute()
                )
                rows = resp.data
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < limit:
                    break
                offset += limit
            except Exception as e:
                logger.error(f'Supabase error: {e}')
                break

    if not all_rows:
        return pd.DataFrame(columns=['symbol', 'date', 'close'])

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


def calc_daily_return_for_date(prices_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Given a long-format price DataFrame, compute the 1-day return for
    each symbol on ``target_date`` relative to the previous available trading day.

    Returns a DataFrame indexed by symbol with columns [close, prev_close, return_1d_pct].
    """
    if prices_df.empty:
        return pd.DataFrame()

    df = prices_df.sort_values(['symbol', 'date'])

    # Find the previous available date across the full dataset
    dates_before = df[df['date'] < target_date]['date'].unique()
    if len(dates_before) == 0:
        return pd.DataFrame()
    prev_date = pd.Timestamp(max(dates_before))

    today_df = (
        df[df['date'] == target_date]
        .set_index('symbol')[['close']]
        .rename(columns={'close': 'close'})
    )
    prev_df = (
        df[df['date'] == prev_date]
        .set_index('symbol')[['close']]
        .rename(columns={'close': 'prev_close'})
    )

    merged = today_df.join(prev_df, how='inner')
    merged['return_1d_pct'] = (merged['close'] / merged['prev_close'] - 1) * 100
    return merged


# ── Sidebar — market selector ─────────────────────────────────────────────── #

stock_select = st.sidebar.radio(
    'Market',
    ['Indonesian Stock (JKSE)', 'S&P 500 (US)'],
    horizontal=False,
    key='mw_sl',
)

sl = 'JKSE' if stock_select == 'Indonesian Stock (JKSE)' else 'SP500'
redis_key = 'div_score_jkse' if sl == 'JKSE' else 'div_score_sp500'


# ── Date picker ───────────────────────────────────────────────────────────── #

today = date.today()
# Default to most recent weekday
default_date = today if today.weekday() < 5 else today - timedelta(days=today.weekday() - 4)

ctrl_cols = st.columns([2, 2, 2, 2, 1])

selected_date = ctrl_cols[0].date_input(
    '📅 Select Date',
    value=default_date,
    max_value=today,
    min_value=date(2015, 1, 1),
    key='mw_date',
)

# ── Load universe ─────────────────────────────────────────────────────────── #

universe_df = get_stock_universe(redis_key)

if universe_df.empty:
    st.error('Could not load stock universe. Please check the data pipeline.', icon='🚨')
    st.stop()

# Keep only needed columns
_KEEP = [c for c in ['sector', 'industry', 'mktCap', 'yield', 'medianProfitMargin',
                      'revenueTTM', 'earningTTM', 'peRatio', 'psRatio', 'revenueGrowth'] if c in universe_df.columns]
universe_df = universe_df[_KEEP].copy()
universe_df['mktCap_B'] = universe_df['mktCap'] / 1_000_000_000


# ── Treemap controls ──────────────────────────────────────────────────────── #

size_var = ctrl_cols[1].selectbox(
    'Size by',
    options=['Market Cap', 'Revenue', 'Net Income'],
    key='mw_size',
)

sector_filter = ctrl_cols[2].selectbox(
    'Sector',
    options=['ALL'] + sorted(universe_df['sector'].dropna().unique().tolist()),
    key='mw_sector',
)

group_secs = ctrl_cols[3].toggle('Group by Sector', value=True, key='mw_group')

min_mcap_b = ctrl_cols[4].number_input(
    'Min MCap (B)',
    min_value=0,
    max_value=1_000,
    value=1 if sl == 'SP500' else 10,
    step=1,
    key='mw_mcap',
    help='Minimum market cap in Billions to include in the treemap'
)

# ── Filter universe ───────────────────────────────────────────────────────── #

filtered_uni = universe_df[universe_df['mktCap_B'] >= min_mcap_b].copy()
if sector_filter != 'ALL':
    filtered_uni = filtered_uni[filtered_uni['sector'] == sector_filter]

symbols_tuple = tuple(sorted(filtered_uni.index.tolist()))

# ── Fetch prices for target date + a few days back ────────────────────────── #

date_to_str   = selected_date.strftime('%Y-%m-%d')
date_from_str = (selected_date - timedelta(days=7)).strftime('%Y-%m-%d')

with st.spinner(f'Fetching price data for {date_to_str}…'):
    prices_df = get_prices_for_date_range(symbols_tuple, date_from_str, date_to_str)

target_ts = pd.Timestamp(selected_date)
returns_df = calc_daily_return_for_date(prices_df, target_ts)

# ── KPI row ───────────────────────────────────────────────────────────────── #

if returns_df.empty:
    st.warning(
        f'No price data found for **{date_to_str}**. '
        'This might be a weekend, public holiday, or data for this date is not yet available.',
        icon='⚠️'
    )
else:
    n_gainers  = int((returns_df['return_1d_pct'] > 0).sum())
    n_losers   = int((returns_df['return_1d_pct'] < 0).sum())
    n_flat     = int((returns_df['return_1d_pct'] == 0).sum())
    avg_return = float(returns_df['return_1d_pct'].mean())
    med_return = float(returns_df['return_1d_pct'].median())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('📅 Date',           date_to_str)
    k2.metric('📈 Gainers',        f'{n_gainers}',
              delta=f'+{n_gainers}', delta_color='normal')
    k3.metric('📉 Losers',         f'{n_losers}',
              delta=f'-{n_losers}', delta_color='inverse')
    k4.metric('↔️ Flat',            f'{n_flat}')
    k5.metric('📊 Avg Daily Return', f'{avg_return:+.2f}%',
              delta=f'Median {med_return:+.2f}%',
              delta_color='normal' if med_return >= 0 else 'inverse')

    # ── Merge returns into universe ───────────────────────────────────────── #

    df_tree = filtered_uni.copy()
    df_tree = df_tree.join(returns_df[['return_1d_pct']], how='left')
    df_tree['return_1d_pct'] = df_tree['return_1d_pct'].fillna(0)

    # Map size variable
    size_col_map = {
        'Market Cap': 'mktCap_B',
        'Revenue':    'revenueTTM',
        'Net Income': 'earningTTM',
    }
    size_col = size_col_map[size_var]

    # Drop rows where size column is missing or <= 0
    df_tree = df_tree[df_tree[size_col].notna() & (df_tree[size_col] > 0)]

    tree_input = pd.DataFrame({
        'sector':       df_tree['sector'],
        'industry':     df_tree['industry'],
        size_var:       df_tree[size_col],
        '1D Return %':  df_tree['return_1d_pct'],
    }, index=df_tree.index).dropna()

    # ── Build treemap ─────────────────────────────────────────────────────── #

    color_threshold = [-10, -5, -2, -0.5, 0.5, 2, 5, 10]

    tree_data = hd.prep_treemap(
        tree_input,
        size_var=size_var,
        color_var='1D Return %',
        color_threshold=color_threshold,
        add_label='color_var',
        group_secs=group_secs,
    )

    option = hp.plot_treemap(
        tree_data,
        size_var=size_var,
        color_var='1D Return %',
        show_gradient=True,
        colormap='red_green',
        group_secs=group_secs,
    )

    click_event_js = "function(params){if(!params.data.children){return params.name;} return null;}"

    clicked_item = st_echarts(
        option,
        events={'click': click_event_js},
        height='860px',
        width='100%',
        key='mw_treemap',
    )

    # ── Tooltip on click: show return details ─────────────────────────────── #

    if clicked_item:
        sym = clicked_item.split()[0]
        if sym in returns_df.index:
            row = returns_df.loc[sym]
            ret = row['return_1d_pct']
            close = row['close']
            prev  = row['prev_close']
            color = '🟢' if ret > 0 else ('🔴' if ret < 0 else '⚪')
            st.info(
                f"{color} **{sym}** — Close: **{close:,.2f}**  |  Prev Close: **{prev:,.2f}**  |  "
                f"1D Return: **{ret:+.2f}%**",
                icon='ℹ️'
            )

    # ── Distribution chart ────────────────────────────────────────────────── #

    with st.expander('📊 Return Distribution', expanded=False):
        import altair as alt

        dist_df = returns_df.reset_index(names='stock')[['return_1d_pct']].dropna()
        dist_df = dist_df.rename(columns={'return_1d_pct': 'return'})

        if dist_df.empty:
            st.info('No return data to plot.')
        else:
            # Pre-filter to p2–p98 range so Altair bins only visible data.
            # Using scale domain alone clips the display but bins are still
            # computed on the full range, causing bars to fall outside the view.
            p2  = float(dist_df['return'].quantile(0.02))
            p98 = float(dist_df['return'].quantile(0.98))
            if p2 >= p98:   # all values identical (e.g. 0% on a non-trading day)
                p2  = float(dist_df['return'].min()) - 0.01
                p98 = float(dist_df['return'].max()) + 0.01

            clipped_df = dist_df[(dist_df['return'] >= p2) & (dist_df['return'] <= p98)].copy()

            zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                color='white', strokeWidth=1.5, strokeDash=[4, 3], opacity=0.7
            ).encode(x=alt.X('x:Q', scale=alt.Scale(domain=[p2, p98])))

            hist = alt.Chart(clipped_df).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                x=alt.X('return:Q', bin=alt.Bin(maxbins=50), title='1D Return (%)'),
                y=alt.Y('count()', title='# Stocks'),
                color=alt.condition(
                    alt.datum['return'] >= 0,
                    alt.value('#4caf50'),
                    alt.value('#d32f2f')
                ),
                tooltip=[
                    alt.Tooltip('return:Q', bin=True, title='Return range', format='.2f'),
                    alt.Tooltip('count()', title='# Stocks'),
                ]
            ).properties(height=260)

            st.altair_chart((hist + zero_line), width='stretch')

    # ── Top gainers / losers table ────────────────────────────────────────── #

    with st.expander('🏆 Top Gainers & Losers', expanded=False):
        merged_tbl = filtered_uni[['sector', 'industry', 'mktCap_B']].join(
            returns_df[['close', 'prev_close', 'return_1d_pct']], how='inner'
        ).dropna(subset=['return_1d_pct'])

        col_g, col_l = st.columns(2)

        top_gainers = merged_tbl.nlargest(20, 'return_1d_pct').reset_index(names='stock')
        top_losers  = merged_tbl.nsmallest(20, 'return_1d_pct').reset_index(names='stock')

        cfig_g = {
            'stock': st.column_config.TextColumn('Stock'),
            'sector': st.column_config.TextColumn('Sector'),
            'close': st.column_config.NumberColumn('Close', format='%,.2f'),
            'prev_close': st.column_config.NumberColumn('Prev Close', format='%,.2f'),
            'return_1d_pct': st.column_config.NumberColumn('1D Return %', format='%+.2f%%'),
            'mktCap_B': st.column_config.NumberColumn('MCap (B)', format='%,.1f'),
            'industry': None,
        }

        with col_g:
            st.markdown('#### 🟢 Top 20 Gainers')
            st.dataframe(
                top_gainers[['stock', 'sector', 'close', 'prev_close', 'return_1d_pct', 'mktCap_B']],
                column_config=cfig_g,
                hide_index=True,
            )

        with col_l:
            st.markdown('#### 🔴 Top 20 Losers')
            st.dataframe(
                top_losers[['stock', 'sector', 'close', 'prev_close', 'return_1d_pct', 'mktCap_B']],
                column_config=cfig_g,
                hide_index=True,
            )
