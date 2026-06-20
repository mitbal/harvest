import io
import os
import json
import logging
from datetime import datetime

import calendar
import redis
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from st_vortree import st_vortree
from st_supabase_connection import SupabaseConnection

import harvest.plot as hp
import harvest.data as hd
from harvest.utils import setup_logging

# Must be the very first Streamlit command
st.set_page_config(page_title='Portfolio Analytics - Panen Dividen', layout='wide')

REQUIRED_COLUMNS = {'Symbol', 'Available Lot', 'Average Price'}
PASTE_RAW_FIELDS_PER_ROW = 11  # expected tokens per stock row in raw paste format


@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger

logger = get_logger('porto')


@st.cache_resource(show_spinner=False)
def get_db_connection() -> SupabaseConnection:
    """
    Establish and cache a connection to the Supabase database.

    Returns:
        SupabaseConnection: Authenticated connection to Supabase
    """
    conn = st.connection("supabase", type=SupabaseConnection)
    conn.auth.sign_in_with_password(
        {
            "email": st.secrets["connections"]["supabase"]["EMAIL_ADDRESS"],
            "password": st.secrets["connections"]["supabase"]["PASSWORD"],
        }
    )
    print('connection to supabase established')
    return conn


def get_user_portfolio(conn: SupabaseConnection, user_email: str) -> dict:
    user_in_db = conn.table("users").select("portfolio").eq("email", user_email).execute()
    if len(user_in_db.data) > 0:
        logger.info(f'portfolio found for {user_email}')
        return user_in_db.data[0]['portfolio']
    else:
        logger.info(f'{user_email} has no portfolio saved in db')
        return None


def update_user_portfolio(conn: SupabaseConnection, portfolio: dict, user_email: str) -> None:

    conn.table("users").upsert(
        {"email": user_email, "portfolio": portfolio, 'modified_at': datetime.now().isoformat()},
        on_conflict="email",
    ).execute()
    logger.info(f'portfolio updated for {user_email}')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r


# --- UI Styling ---
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #064E3B 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 1rem;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Premium KPI Card Styling */
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        backdrop-filter: blur(10px);
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(5, 150, 105, 0.3);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    
    .kpi-delta {
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .delta-plus { color: #059669; }
    .delta-minus { color: #DC2626; }
    
    /* Custom container for cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
</style>
""")

def render_kpi(label, value, delta=None, delta_type="normal"):
    delta_html = ""
    if delta:
        cls = "delta-plus" if (delta_type == "normal" and "-" not in delta) or (delta_type == "inverse" and "-" in delta) else "delta-minus"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    
    st.html(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """)

conn = get_db_connection()

data_input_expand_flag = True
if 'porto_df' not in st.session_state:
    if st.user.is_logged_in:
        data = get_user_portfolio(conn, st.user.email)
        print(st.user.email, data)
        if data is None:
            st.session_state['porto_df'] = None
        elif len(data) > 0:
            st.session_state['porto_df'] = pd.DataFrame(data)
        else:
            st.session_state['porto_df'] = None
    else:
        st.session_state['porto_df'] = None
else:
    data_input_expand_flag = False


# --- Header ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title('Portfolio Analytics')
    if st.user.is_logged_in:
        st.markdown(f"**Welcome back, {st.user.name.split()[0]}!** Here's your harvest overview for today.")
    else:
        st.markdown("Analyze your portfolio performance and dividend growth.")

with col_head2:
    if st.user.is_logged_in:
         st.button('Log Out', icon=':material/logout:', on_click=st.logout, width='stretch')
         if st.session_state.get('porto_df') is not None:
             if st.button('💾 Sync Portfolio to Cloud', width='stretch'):
                 update_user_portfolio(conn, st.session_state['porto_df'].to_dict(), st.user.email)
                 st.success('Portfolio synced successfully!', icon="✅")
    else:
         st.button('Log in with Google', icon=':material/login:', on_click=lambda: st.login('google'), width='stretch')


with st.expander('📥 Porto Data Input', expanded=data_input_expand_flag):

    input_cols = st.columns([1, 2])
    
    with input_cols[0]:
        st.markdown("### Selection")
        method = st.radio('Import Method', ['Upload CSV', 'Form', 'Paste Raw'], index=1, horizontal=False)
        
        st.divider()
        st.markdown("### Settings")
        target = st.number_input(
            label='Target Annual Income (M IDR)',
            value=240, step=1, min_value=1, max_value=10_000,
            format='%d',
            help="Your financial freedom goal"
        )

        baseline = st.number_input(
            label='Benchmark (%)',
            value=6.35, step=.01, min_value=0.01, max_value=99.99,
            help="Benchmark yield for comparison (e.g. S&P500 or Govt Bond)"
        )

    with input_cols[1]:
        st.markdown(f"### {method} Interface")
        with st.form('abc', border=False):
            if method == 'Upload CSV':
                uploaded_file = st.file_uploader('Select your portfolio CSV', type='csv')
                if uploaded_file:
                    st.session_state['porto_file'] = uploaded_file
                
            elif method == 'Paste Raw':
                raw = st.text_area('Paste data from Stockbit Portfolio page', height=200)
            
            elif method == 'Form':
                if st.session_state['porto_df'] is None:
                    # Load default template from the data folder
                    try:
                        example_df = pd.read_csv('data/porto_sample1.csv')
                    except Exception:
                        example_df = pd.DataFrame(columns=['Symbol', 'Available Lot', 'Average Price'])
                        example_df.loc[0] = ['BBCA', 10, 10000]
                else:
                    example_df = st.session_state['porto_df'].copy(deep=True)
                
                example_df = example_df.reset_index(drop=True)
                edited_df = st.data_editor(example_df, num_rows='dynamic', hide_index=True, width='stretch')

            submit = st.form_submit_button('🚀 Load Portfolio Data', width='stretch')
            
            if submit:
                if method == 'Upload CSV':
                    porto_file = st.session_state.get('porto_file')
                    if porto_file is None or porto_file == 'EMPTY':
                        st.error('⚠️ Please select a CSV file before submitting.', icon="📂")
                    else:
                        try:
                            porto_file.seek(0)
                            uploaded_df = pd.read_csv(porto_file, sep=',', dtype='str')
                            missing_cols = REQUIRED_COLUMNS - set(uploaded_df.columns)
                            if missing_cols:
                                st.error(f'❌ CSV is missing required columns: **{missing_cols}**. Expected: Symbol, Available Lot, Average Price', icon="🗂️")
                            elif uploaded_df.empty:
                                st.error('❌ The uploaded CSV file is empty.', icon="📭")
                            else:
                                st.session_state['porto_df'] = uploaded_df
                        except Exception as e:
                            st.error(f'❌ Failed to read CSV file: {e}', icon="🚫")
                            logger.exception('CSV upload parsing failed')

                elif method == 'Paste Raw':
                    if not raw or not raw.strip():
                        st.error('⚠️ Please paste your portfolio data before submitting.', icon="📋")
                    else:
                        try:
                            rows = np.array(raw.split())
                            if len(rows) % PASTE_RAW_FIELDS_PER_ROW != 0:
                                st.error(
                                    f'❌ Pasted data has {len(rows)} tokens, which is not a multiple of {PASTE_RAW_FIELDS_PER_ROW}. '
                                    'Please make sure you copied the full table from Stockbit.',
                                    icon="⚠️"
                                )
                            else:
                                stock = rows[range(0, len(rows), PASTE_RAW_FIELDS_PER_ROW)]
                                lot = [x.replace(',', '') for x in rows[range(1, len(rows), PASTE_RAW_FIELDS_PER_ROW)]]
                                price = [p.replace(',', '') for p in rows[range(3, len(rows), PASTE_RAW_FIELDS_PER_ROW)]]

                                # Validate numeric fields
                                bad_lots = [v for v in lot if not _is_positive_number(v)]
                                bad_prices = [v for v in price if not _is_positive_number(v)]
                                if bad_lots:
                                    st.error(f'❌ Non-numeric or negative lot values detected: {bad_lots}', icon="🔢")
                                elif bad_prices:
                                    st.error(f'❌ Non-numeric or negative price values detected: {bad_prices}', icon="🔢")
                                else:
                                    df = pd.DataFrame({
                                        'Symbol': stock,
                                        'Available Lot': lot,
                                        'Average Price': price
                                    })
                                    st.session_state['porto_df'] = df
                        except Exception as e:
                            st.error(f'❌ Failed to parse pasted data: {e}', icon="🚫")
                            logger.exception('Paste Raw parsing failed')

                elif method == 'Paste CSV':
                    if not raw or not raw.strip():
                        st.error('⚠️ Please paste your CSV data before submitting.', icon="📋")
                    else:
                        try:
                            input_str = io.StringIO(raw)
                            pasted_df = pd.read_csv(input_str, sep=';', dtype='str')
                            missing_cols = REQUIRED_COLUMNS - set(pasted_df.columns)
                            if missing_cols:
                                st.error(f'❌ Pasted CSV is missing required columns: **{missing_cols}**', icon="🗂️")
                            elif pasted_df.empty:
                                st.error('❌ Pasted CSV data is empty.', icon="📭")
                            else:
                                st.session_state['porto_df'] = pasted_df
                        except Exception as e:
                            st.error(f'❌ Failed to parse pasted CSV: {e}', icon="🚫")
                            logger.exception('Paste CSV parsing failed')

                elif method == 'Form':
                    df = edited_df.copy(deep=True)
                    df.dropna(subset=['Symbol'], inplace=True)
                    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
                    df = df[df['Symbol'] != '']

                    if df.empty:
                        st.error('❌ No valid rows in the form. Please add at least one stock.', icon="📭")
                    else:
                        # Validate numeric columns
                        invalid_lots = df[pd.to_numeric(df['Available Lot'], errors='coerce').isna() | (pd.to_numeric(df['Available Lot'], errors='coerce') <= 0)]
                        invalid_prices = df[pd.to_numeric(df['Average Price'], errors='coerce').isna() | (pd.to_numeric(df['Average Price'], errors='coerce') <= 0)]
                        if not invalid_lots.empty:
                            st.error(f'❌ Invalid or non-positive lot values in rows: {invalid_lots["Symbol"].tolist()}', icon="🔢")
                        elif not invalid_prices.empty:
                            st.error(f'❌ Invalid or non-positive price values in rows: {invalid_prices["Symbol"].tolist()}', icon="🔢")
                        else:
                            st.session_state['porto_df'] = df

                if st.session_state.get('porto_df') is not None:
                    logger.info(f'Porto data submitted via {method}')
                    logger.info(f'target: {target}. baseline: {baseline}. porto: {st.session_state["porto_df"].to_records()}')


api_key = os.environ.get('FMP_API_KEY', '')


def _is_positive_number(val: str) -> bool:
    """Return True if val can be parsed as a positive float."""
    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False


@st.cache_data(ttl=60*60)
def get_company_profile_data(porto):

    redis_url = os.environ.get('REDIS_URL')
    if not redis_url:
        raise EnvironmentError('REDIS_URL environment variable is not set.')

    try:
        r = connect_redis(redis_url)
        rjson = r.get('div_score_jkse')
    except Exception as e:
        raise ConnectionError(f'Failed to connect to Redis or fetch data: {e}') from e

    if rjson is None:
        raise ValueError('No data found in Redis for key "div_score_jkse". Please ensure the data pipeline has run.')

    try:
        if isinstance(rjson, bytes) and rjson.startswith(b'PAR1'):
            cp_df = pd.read_parquet(io.BytesIO(rjson))
        else:
            div_score_json = json.loads(rjson)
            if 'content' in div_score_json:
                cp_df = pd.DataFrame(json.loads(div_score_json['content']))
            else:
                cp_df = pd.DataFrame(div_score_json)
    except Exception as e:
        raise ValueError(f'Failed to parse company profile data from Redis: {e}') from e

    cp_df.rename(columns={'symbol': 'stock'}, inplace=True)
    if 'stock' in cp_df.columns:
        cp_df.set_index('stock', inplace=True)

    for col in ['price', 'sector', 'lastDiv']:
        if col not in cp_df.columns:
            raise KeyError(f'Expected column "{col}" not found in company profile data.')

    cp_df['Symbol'] = [x[:-3] for x in cp_df.index.to_list()]
    df = porto.merge(cp_df[['Symbol', 'price', 'sector', 'lastDiv']])
    df.rename(columns={'lastDiv': 'div_rate', 'price': 'last_price'}, inplace=True)

    if df.empty:
        unknown = set(porto['Symbol'].tolist()) - set(cp_df['Symbol'].tolist())
        raise ValueError(
            f'No matching stocks found after merging with company profile data. '
            f'Symbols not found in database: {unknown}'
        )

    return df


@st.cache_data(ttl=60*60)
def get_dividend_data(porto):
    stock_list = [x+'.JK' for x in porto['Symbol']]
    divs = {}
    for stock in stock_list:
        div_df = hd.get_dividend_history_single_stock_dag(stock)
        if div_df is not None:
            div_df = div_df[div_df['dividend_type'] != 'special']
            divs[stock] = div_df
        else:
            logger.info(f'stock {stock} do not have dividend history')
    return divs


if st.session_state.get('porto_df') is None:
    st.info('👆 Please upload or enter your portfolio data above to get started.', icon="📊")
    st.stop()

st.session_state['porto_df'].dropna(how='all', inplace=True)
st.session_state['porto_df'].dropna(subset=['Symbol'], inplace=True)

if st.session_state['porto_df'].empty:
    st.error('❌ Your portfolio data has no valid rows. Please check your input.', icon="📭")
    st.stop()

try:
    df = get_company_profile_data(st.session_state['porto_df'])
except (ConnectionError, EnvironmentError) as e:
    st.error(f'🔌 **Connection Error:** {e}', icon="🔌")
    logger.exception('Failed to connect to data source')
    st.stop()
except ValueError as e:
    st.error(f'📉 **Data Error:** {e}', icon="⚠️")
    logger.exception('Data error in company profile fetch')
    st.stop()
except KeyError as e:
    st.error(f'🗂️ **Schema Error:** Missing expected column {e}', icon="🗂️")
    logger.exception('Schema mismatch in company profile data')
    st.stop()
except Exception as e:
    st.error(f'❌ **Unexpected error loading company profile data:** {e}', icon="🚫")
    logger.exception('Unexpected error in get_company_profile_data')
    st.stop()

try:
    divs = get_dividend_data(st.session_state['porto_df'])
except Exception as e:
    st.warning(f'⚠️ Could not load dividend history: {e}. Some features may be limited.', icon="📅")
    logger.exception('Failed to load dividend data')
    divs = {}

# Cast and validate numeric columns
try:
    df['current_lot'] = pd.to_numeric(df['Available Lot'], errors='raise').astype(float)
except (ValueError, TypeError) as e:
    st.error(f'❌ **Invalid lot values in portfolio:** {e}', icon="🔢")
    st.stop()

try:
    df['avg_price'] = pd.to_numeric(df['Average Price'], errors='raise').astype(float)
except (ValueError, TypeError) as e:
    st.error(f'❌ **Invalid price values in portfolio:** {e}', icon="🔢")
    st.stop()

if (df['current_lot'] <= 0).any():
    bad = df[df['current_lot'] <= 0]['Symbol'].tolist()
    st.error(f'❌ Lot values must be positive. Check: {bad}', icon="🔢")
    st.stop()

if (df['avg_price'] <= 0).any():
    bad = df[df['avg_price'] <= 0]['Symbol'].tolist()
    st.error(f'❌ Average price values must be positive. Check: {bad}', icon="🔢")
    st.stop()

df['total_invested'] = df['current_lot'] * df['avg_price'] * 100
df['yield_on_cost'] = df['div_rate'] / df['avg_price'] * 100
df['yield_on_price'] = df['div_rate'] / df['last_price'] * 100
df['total_dividend'] = (df['div_rate'] * df['current_lot'] * 100).astype(int)

annual_dividend = df['total_dividend'].sum()
total_investment = df['total_invested'].sum()
current_investment_value = (df['current_lot'] * df['last_price'] * 100).sum()
achieve_percentage = annual_dividend / target * 100 / 1_000_000 if target > 0 else 0
total_yield_on_cost = annual_dividend / total_investment * 100 if total_investment > 0 else 0

df_display = df[['Symbol', 'Available Lot', 'avg_price', 'total_invested', 'div_rate', 'last_price', 
                 'yield_on_cost', 'yield_on_price', 'total_dividend']].copy(deep=True)


# Overall summary
with st.container(border=True):
    overall_cols = st.columns(5)
    
    # 1. Total Dividend Yield on Cost
    delta_val = total_yield_on_cost - baseline
    delta_text = f"{delta_val:+.2f}% vs benchmark"
    with overall_cols[0]:
        render_kpi("Yield on Cost", f"{total_yield_on_cost:.2f}%", delta_text)
    
    # 2. Dividend Annual Income
    with overall_cols[1]:
        render_kpi("Annual Income", f"IDR {annual_dividend:,.0f}")
        
    # 3. Total Investment
    with overall_cols[2]:
        render_kpi("Total Invested", f"IDR {total_investment:,.0f}")
        
    # 4. Current Market Value
    market_delta = current_investment_value - total_investment
    market_delta_text = f"IDR {market_delta:+,.0f}"
    with overall_cols[3]:
        render_kpi("Market Value", f"IDR {current_investment_value:,.0f}", market_delta_text)
        
    # 5. Percent on Target
    with overall_cols[4]:
        render_kpi("Target Progress", f"{achieve_percentage:.2f}%")

# Table List
with st.container(border=True):

    tabs = st.tabs(['Table View', 'Bar Chart View', 'Voronoi Treemap'])
    
    with tabs[0]:
        st.subheader('Portfolio Holdings', divider='grey')

        cfig = {
            'yield_on_cost': st.column_config.NumberColumn(
                'Yield on Cost',
                format='%.2f%%',
                help='Dividend Yield based on your Average Purchase Price'
            ),
            'yield_on_price': st.column_config.NumberColumn(
                'Yield on Price',
                format='%.2f%%',
                help='Dividend Yield based on Current Market Price'
            ),
            'div_rate': st.column_config.NumberColumn(
                'Last Dividend',
                format='IDR %,.0f'
            ),
            'avg_price': st.column_config.NumberColumn(
                'Avg Price',
                format='IDR %,d'
            ),
            'total_invested': st.column_config.NumberColumn(
                'Total Invested',
                format='IDR %,d'
            ),
            'last_price': st.column_config.NumberColumn(
                'Market Price',
                format='IDR %,d'
            ),
            'total_dividend': st.column_config.NumberColumn(
                'Annual Dividend',
                format='IDR %,d'
            ),
            'Available Lot': st.column_config.NumberColumn(
                'Lots',
                format='%,d'
            )
        }

        main_event = st.dataframe(
            df_display.set_index('Symbol'),
            on_select='rerun',
            selection_mode='single-row',
            column_config=cfig,
            width='stretch'
        )

    with tabs[1]:
        div_bar = alt.Chart(df_display).mark_bar().encode(
            x=alt.X('Symbol'),
            y=alt.Y('total_dividend')
        )
        yield_bar = alt.Chart(df_display).mark_line(color='orange').encode(
            x=alt.X('Symbol'),
            y=alt.Y('yield_on_cost', scale=alt.Scale(domain=[0, 100])),
        )
        combined_chart = (div_bar + yield_bar).resolve_scale(y='independent')
        st.altair_chart(combined_chart, width="stretch")

    with tabs[2]:
        ctrl_cols = st.columns([2, 2, 1, 1, 2])
        value_metric = ctrl_cols[0].selectbox(
            'Value metric',
            options=['total_invested', 'total_dividend'],
            format_func=lambda x: 'Total Invested' if x == 'total_invested' else 'Total Dividend',
            key='vortree_metric'
        )
        color_scheme = ctrl_cols[1].selectbox(
            'Color scheme',
            ['tableau10', 'category10', 'pastel1'],
            key='vortree_color'
        )
        show_values = ctrl_cols[2].checkbox('Show values', value=False, key='vortree_show_values')
        show_pct_only = ctrl_cols[3].checkbox('Show % only', value=True, key='vortree_pct_only')
        treemap_height = ctrl_cols[4].slider('Size', 300, 900, 500, key='vortree_height')

        ctrl_cols2 = st.columns([2, 2, 2, 2])
        border_color = ctrl_cols2[0].color_picker('Border color', value='#000000', key='vortree_border_color')
        label_scale = ctrl_cols2[1].number_input('Label scale', min_value=0.1, max_value=3.0, value=1.5, step=0.1, key='vortree_label_scale')
        
        if 'vortree_refresh_count' not in st.session_state:
            st.session_state['vortree_refresh_count'] = 0

        if ctrl_cols2[2].button('Refresh Plot', icon=':material/refresh:', width='stretch'):
            st.session_state['vortree_refresh_count'] += 1
            st.rerun()
        
        treemap_df = df_display[['Symbol', value_metric]].copy()
        treemap_df['sector'] = df['sector'].values
        st_vortree(
            treemap_df,
            name_col='Symbol',
            value_col=value_metric,
            group_col='sector',
            color_scheme=color_scheme,
            show_values=show_values,
            show_pct_only=show_pct_only,
            label_scale=label_scale,
            border_color=border_color,
            border_width=2,
            show_legend=True,
            height=treemap_height,
            key=f'porto_vortree_{st.session_state["vortree_refresh_count"]}'
        )


if main_event.selection.get('rows'):

    symbol = df_display.iloc[main_event.selection['rows'][0]]['Symbol']

    with st.expander('Dividend History', expanded=True):

        if not divs:
            st.warning('⚠️ Dividend history could not be loaded. Please check your connection.', icon="📅")
        elif symbol+'.JK' not in divs.keys():
            st.info(f'No dividend history available for **{symbol}**.', icon="📭")
        else:
            try:
                div_df = pd.DataFrame(divs[symbol+'.JK'])

                if div_df.empty or 'date' not in div_df.columns or 'adjDividend' not in div_df.columns:
                    st.info(f'Dividend history for **{symbol}** has no payable records.', icon="📭")
                else:
                    div_hist_cols = st.columns([3, 10, 5])
                    with div_hist_cols[0]:
                        st.dataframe(
                            div_df[['date', 'adjDividend']],
                            column_config={
                                'date': st.column_config.DateColumn('Ex-Date'),
                                'adjDividend': st.column_config.NumberColumn('Dividend', format='%,.1f')
                            },
                            height=420,
                            hide_index=True
                        )

                    with div_hist_cols[1]:
                        try:
                            stats = hd.calc_div_stats(hd.preprocess_div(div_df))
                            div_bar = hp.plot_dividend_history(
                                div_df,
                                extrapolote=True,
                                n_future_years=5,
                                last_val=df_display.iloc[main_event.selection['rows'][0]]['div_rate'],
                                inc_val=stats['historical_mean_flat']
                            )
                            st.altair_chart(div_bar, width="stretch")
                        except Exception as e:
                            st.warning(f'⚠️ Could not render dividend history chart: {e}', icon="📊")
                            logger.exception(f'Dividend history chart failed for {symbol}')
            except Exception as e:
                st.error(f'❌ Failed to display dividend history for {symbol}: {e}', icon="🚫")
                logger.exception(f'Dividend history display error for {symbol}')

            # with div_hist_cols[2]:


with st.expander('📊 Sectoral Exposure', expanded=True):

    sector_cols = st.columns([1, 1, 1])    
    with sector_cols[0]:
        st.markdown("**Dividends by Sector**")
        sector_df = df.groupby('sector')['total_dividend'].sum().to_frame().sort_values('total_dividend', ascending=False).reset_index()
        event = st.dataframe(
            sector_df,
            selection_mode=['single-row'],
            on_select='rerun',
            hide_index=True,
            key='sector_table',
            width='stretch',
            column_config={
                'sector': 'Sector',
                'total_dividend': st.column_config.NumberColumn('Total Div (IDR)', format='%,d'),
            }
        )

    with sector_cols[1]:
        st.markdown("**Stocks in Selection**")
        if len(event.selection['rows']) > 0:
            row_idx = event.selection['rows'][0]
            sector_name = sector_df.loc[row_idx, 'sector']
            st.dataframe(
                df[df['sector'] == sector_name][['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False), 
                hide_index=True, 
                width='stretch',
                column_config={
                    'total_dividend': st.column_config.NumberColumn('Total Div (IDR)', format='%,d'),
                }
            )
        else:
            st.info('Select a sector on the left to see holdings', icon="👈")

    with sector_cols[2]:
        st.markdown("**Diversification**")
        sector_pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta='sum(total_dividend)',
            color=alt.Color('sector', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=['sector', alt.Tooltip('sum(total_dividend)', format=',d')]
        ).properties(height=250)
        st.altair_chart(sector_pie, width="stretch")


with st.expander('📅 Dividend Timeline', expanded=True):

    view_cols = st.columns([2, 1])
    with view_cols[0]:
        view_type = st.segmented_control('View Pattern', ['Calendar', 'Monthly Bar', 'Grid Table'], default='Monthly Bar')

    # prepare calendar data
    div_lists = []
    for index, row in df.iterrows():

        r = row.to_dict()
        stock = r['Symbol']+'.JK'
        if stock not in divs.keys():
            continue

        try:
            div_df = pd.DataFrame(divs[stock])
            if div_df.empty or 'date' not in div_df.columns:
                continue

            div_df['year'] = div_df['date'].apply(lambda x: x.split('-')[0])
            div_df['date'] = pd.to_datetime(div_df['date']).dt.tz_localize(None)

            end_date = pd.Timestamp('today').to_datetime64()
            start_date = (end_date - pd.Timedelta(days=365)).to_datetime64()

            current_year = datetime.today().year
            last_year_div = div_df[(pd.to_datetime(div_df['date']) >= start_date) & (pd.to_datetime(div_df['date']) < end_date)].copy(deep=True)
            last_year_div['Symbol'] = stock
            last_year_div['Lot'] = r['current_lot']
            if r.get('last_price', 0) > 0:
                last_year_div['yield'] = last_year_div['adjDividend'] / r['last_price'] * 100
            else:
                last_year_div['yield'] = 0.0

            if not last_year_div.empty:
                div_lists += [last_year_div]
        except Exception as e:
            logger.warning(f'Skipping dividend timeline entry for {stock}: {e}')
            continue

    if not div_lists:
        st.info('📅 No dividend payment data found in the past 12 months for your portfolio.', icon="📭")
        st.stop()

    all_divs = pd.concat(div_lists, ignore_index=True)
    all_divs['total_dividend'] = (all_divs['Lot'] * all_divs['adjDividend'] * 100).astype('int')
    all_divs['Date'] = pd.to_datetime(all_divs['date']).dt.tz_localize(None)
    # all_divs['new_date'] = all_divs['date'].apply(lambda x: x + pd.Timedelta(days=14))
    all_divs['month'] = all_divs['date'].apply(lambda x: x.month)
    
    month_div = all_divs.groupby('month')['total_dividend'].sum().to_frame().reset_index()
    month_div['month_name'] = month_div['month'].apply(lambda x: calendar.month_name[x])
    
    # st.write(all_divs)
    if view_type == 'Calendar':
        all_divs['date'] = all_divs['Date']
        all_divs['date'] = all_divs['date'].apply(lambda x: x.replace(year=current_year-1))
        all_divs['symbol'] = all_divs['Symbol']
        cal = hp.plot_dividend_calendar(all_divs)
        st.altair_chart(cal, width="stretch")
    
    elif view_type == 'Monthly Bar':
        bar_cols = st.columns([1, 2])
        with bar_cols[0]:
            st.dataframe(
                month_div[['month_name', 'total_dividend']],
                column_config={
                    'month_name': 'Month',
                    'total_dividend': st.column_config.NumberColumn('Total Div (IDR)', format='%,d'),
                },
                hide_index=True,
                width='stretch'
            )

        with bar_cols[1]:
            month_bar = alt.Chart(month_div).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('month_name:N', sort=month_div['month_name'].tolist(), title='Month'),
                y=alt.Y('total_dividend:Q', title='Total Dividend (IDR)'),
                color=alt.value('#10B981'),
                tooltip=['month_name', alt.Tooltip('total_dividend', format=',d')]
            ).properties(height=300)
            st.altair_chart(month_bar, width="stretch")
    
    else:
        # Grid Table View
        row_1 = st.container()
        with row_1:
            row_1_cols = st.columns(6)
            for c, i in zip(row_1_cols, range(1, 7)):
                m = all_divs[all_divs['month'] == i]
                c.markdown(f"**{calendar.month_name[i]}**")
                c.dataframe(
                    m[['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False),
                    hide_index=True,
                    width='stretch',
                    column_config={
                        'total_dividend': st.column_config.NumberColumn('Div', format='%,d'),
                    },
                    height=200
                )

        row_2 = st.container()
        with row_2:
            row_2_cols = st.columns(6)
            for c, i in zip(row_2_cols, range(7, 13)):
                m = all_divs[all_divs['month'] == i]
                c.markdown(f"**{calendar.month_name[i]}**")
                c.dataframe(
                    m[['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False),
                    hide_index=True,
                    width='stretch',
                    column_config={
                        'total_dividend': st.column_config.NumberColumn('Div', format='%,d'),
                    },
                    height=200
                )

# Project future earnings
with st.expander('📈 Compounding Projection', expanded=True):
    st.markdown("Estimate your future returns based on compounding dividends and yield growth.")

    proj_input_cols = st.columns([1, 1, 3])
    with proj_input_cols[0]:
        number_of_year = st.number_input('Years', value=25, min_value=1, max_value=50)
    with proj_input_cols[1]:
        inc = st.number_input('Expected Yield (%)', value=total_yield_on_cost, min_value=0.1, max_value=50.0, step=0.1)
    
    futures = [0]*number_of_year
    for i in range(number_of_year):
        futures[i] = annual_dividend * (1+inc/100)**i

    df_future = pd.DataFrame({'years': [f'Year {i+1:02d}' for i in range(number_of_year)], 'returns': futures})
    df_future['achieved'] = df_future['returns'] > (target*1_000_000)
    df_future['yield'] = df_future['returns'] / total_investment * 100
    
    base_chart = alt.Chart(df_future)
    return_chart = base_chart.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('years:N', title='Compounding Journey'),
        y=alt.Y('returns:Q', title='Annual Income (IDR)'),
        color=alt.condition(alt.datum['achieved'], alt.value('#059669'), alt.value('#93C5FD')),
        tooltip=['years', alt.Tooltip('returns', format=',.0f')]
    )

    yield_chart = base_chart.mark_line(point=True, color='#D97706').encode(
        x=alt.X('years:N'),
        y=alt.Y('yield:Q', title='Yield on Cost (%)'),
        tooltip=['years', alt.Tooltip('yield', format='.2f')]
    )

    future_chart = (return_chart + yield_chart).resolve_scale(y='independent').properties(height=400)
    st.altair_chart(future_chart, width="stretch")



