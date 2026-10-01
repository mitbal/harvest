import io
import os
import json
import html
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
from harvest import chart_style as cs
from harvest.utils import setup_logging

# Must be the very first Streamlit command
st.set_page_config(page_title='Portfolio Analytics - Panen Dividen', layout='wide')
hp.enable_chart_theme()

REQUIRED_COLUMNS = {'Symbol', 'Available Lot', 'Average Price'}
PASTE_RAW_FIELDS_PER_ROW = 11  # expected tokens per stock row in raw paste format
PROJECTION_MAX_GROWTH = 50.0  # upper bound for the income growth input (%)


def _is_positive_number(val: str) -> bool:
    """Return True if val can be parsed as a finite positive float."""
    try:
        number = float(val)
        return np.isfinite(number) and number > 0
    except (ValueError, TypeError):
        return False


def normalize_portfolio(portfolio) -> pd.DataFrame:
    """Validate portfolio input and combine duplicate symbols at weighted cost."""
    df = pd.DataFrame(portfolio).copy(deep=True)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f'Missing required columns: {missing}.')

    df = df[['Symbol', 'Available Lot', 'Average Price']].dropna(how='all')
    df['Symbol'] = (
        df['Symbol']
        .astype('string')
        .str.strip()
        .str.upper()
        .str.removesuffix('.JK')
    )
    df = df[df['Symbol'].notna() & (df['Symbol'] != '')].copy()
    if df.empty:
        raise ValueError('Add at least one stock with a symbol, lot count, and average price.')

    for column in ['Available Lot', 'Average Price']:
        cleaned = df[column].astype('string').str.replace(',', '', regex=False).str.strip()
        df[column] = pd.to_numeric(cleaned, errors='coerce')
        invalid = df[column].isna() | ~np.isfinite(df[column]) | (df[column] <= 0)
        if invalid.any():
            symbols = ', '.join(df.loc[invalid, 'Symbol'].astype(str).tolist())
            raise ValueError(f'{column} must be a positive number. Check: {symbols}.')

    df['_invested'] = df['Available Lot'] * df['Average Price']
    grouped = df.groupby('Symbol', sort=False, as_index=False).agg(
        {'Available Lot': 'sum', '_invested': 'sum'}
    )
    grouped['Average Price'] = grouped['_invested'] / grouped['Available Lot']
    return grouped[['Symbol', 'Available Lot', 'Average Price']]


def parse_raw_portfolio(raw: str) -> pd.DataFrame:
    """Parse a Stockbit portfolio table copied as whitespace-delimited text."""
    if not raw or not raw.strip():
        raise ValueError('Paste your Stockbit portfolio data before loading it.')

    rows = raw.split()
    if len(rows) % PASTE_RAW_FIELDS_PER_ROW != 0:
        raise ValueError(
            f'Pasted data contains {len(rows)} fields; expected a multiple of '
            f'{PASTE_RAW_FIELDS_PER_ROW}. Copy the complete Stockbit table and try again.'
        )

    portfolio = pd.DataFrame({
        'Symbol': rows[0::PASTE_RAW_FIELDS_PER_ROW],
        'Available Lot': rows[1::PASTE_RAW_FIELDS_PER_ROW],
        'Average Price': rows[3::PASTE_RAW_FIELDS_PER_ROW],
    })
    return normalize_portfolio(portfolio)


def portfolio_to_records(portfolio) -> list[dict]:
    """Return a JSON-safe, record-oriented portfolio payload."""
    normalized = normalize_portfolio(portfolio)
    return json.loads(normalized.to_json(orient='records'))


def build_last_year_dividends(holdings: pd.DataFrame, divs: dict, today=None) -> list[pd.DataFrame]:
    """Return each holding's dividend payments from the trailing 12 months, tagged with symbol and lots."""
    end_date = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    start_date = end_date - pd.Timedelta(days=365)

    div_lists = []
    for row in holdings.itertuples(index=False):
        symbol = row.Symbol
        history = divs.get(f'{symbol}.JK')
        if history is None:
            continue

        try:
            div_df = pd.DataFrame(history)
            if div_df.empty or 'date' not in div_df.columns or 'adjDividend' not in div_df.columns:
                continue

            # Accept strings, naive or tz-aware timestamps
            div_df['date'] = pd.to_datetime(div_df['date'], errors='coerce', utc=True).dt.tz_localize(None)
            in_window = (div_df['date'] >= start_date) & (div_df['date'] < end_date)
            last_year_div = div_df[in_window].copy(deep=True)
            if last_year_div.empty:
                continue

            last_year_div['Symbol'] = symbol
            last_year_div['Lot'] = row.current_lot
            last_year_div['yield'] = last_year_div['adjDividend'] / row.last_price * 100
            div_lists.append(last_year_div)
        except Exception as e:
            logger.warning(f'Skipping dividend timeline entry for {symbol}: {e}')

    return div_lists


def project_income(annual_dividend: float, initial_cost: float, growth_pct: float, years: int) -> pd.DataFrame:
    """Project income when all dividends are reinvested and income grows at growth_pct per year.

    Year N is N years from today. Yield is measured against the original cost plus all
    dividends reinvested before that year, since reinvested cash is also invested capital.
    """
    year_numbers = np.arange(1, years + 1)
    returns = annual_dividend * (1 + growth_pct / 100) ** year_numbers
    # Income received in years 0..N-1 has been reinvested by year N
    previous_income = np.concatenate([[annual_dividend], returns[:-1]])
    cost_basis = initial_cost + np.cumsum(previous_income)
    return pd.DataFrame({
        'years': [f'Year {n:02d}' for n in year_numbers],
        'returns': returns,
        'yield': returns / cost_basis * 100,
    })


def render_dividend_timeline(div_lists: list[pd.DataFrame], view_type: str) -> None:
    """Render timeline views without controlling the rest of the page flow."""
    all_divs = pd.concat(div_lists, ignore_index=True)
    all_divs['total_dividend'] = (
        all_divs['Lot'] * all_divs['adjDividend'] * 100
    ).astype('int')
    all_divs['Date'] = pd.to_datetime(all_divs['date']).dt.tz_localize(None)
    all_divs['month'] = all_divs['Date'].dt.month

    month_div = all_divs.groupby('month')['total_dividend'].sum().reset_index()
    month_div['month_name'] = month_div['month'].map(lambda month: calendar.month_name[month])

    if view_type == 'Calendar':
        calendar_year = datetime.today().year - 1

        def normalize_calendar_year(date):
            try:
                return date.replace(year=calendar_year)
            except ValueError:
                return date.replace(year=calendar_year, day=28)

        calendar_df = all_divs.copy(deep=True)
        calendar_df['date'] = calendar_df['Date'].map(normalize_calendar_year)
        calendar_df['symbol'] = calendar_df['Symbol']
        st.altair_chart(hp.plot_dividend_calendar(calendar_df), width='stretch', theme=None)
    elif view_type == 'Monthly Bar':
        bar_cols = st.columns([1, 2])
        with bar_cols[0]:
            st.dataframe(
                month_div[['month_name', 'total_dividend']],
                column_config={
                    'month_name': 'Month',
                    'total_dividend': st.column_config.NumberColumn(
                        'Total Div (IDR)', format='%,d'
                    ),
                },
                hide_index=True,
                width='stretch',
            )

        with bar_cols[1]:
            month_bar = alt.Chart(month_div).mark_bar(
                cornerRadiusTopLeft=5, cornerRadiusTopRight=5
            ).encode(
                x=alt.X('month_name:N', sort=month_div['month_name'].tolist(), title='Month'),
                y=alt.Y('total_dividend:Q', title='Total Dividend (IDR)'),
                color=alt.value(cs.CHART_PRIMARY),
                tooltip=['month_name', alt.Tooltip('total_dividend', format=',d')],
            ).properties(height=300)
            st.altair_chart(month_bar, width='stretch', theme=None)
    else:
        for first_month in (1, 7):
            month_cols = st.columns(6)
            for column, month in zip(month_cols, range(first_month, first_month + 6)):
                monthly_payments = all_divs[all_divs['month'] == month]
                column.markdown(f'**{calendar.month_name[month]}**')
                column.dataframe(
                    monthly_payments[['Symbol', 'total_dividend']].sort_values(
                        'total_dividend', ascending=False
                    ),
                    hide_index=True,
                    width='stretch',
                    column_config={
                        'total_dividend': st.column_config.NumberColumn('Div', format='%,d'),
                    },
                    height=200,
                )


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
    logger.info('connection to supabase established')
    return conn


def get_user_portfolio(conn: SupabaseConnection, user_email: str) -> dict:
    user_in_db = conn.table("users").select("portfolio").eq("email", user_email).execute()
    if len(user_in_db.data) > 0:
        logger.info('portfolio found in db')
        return user_in_db.data[0]['portfolio']
    else:
        logger.info('user has no portfolio saved in db')
        return None


def update_user_portfolio(conn: SupabaseConnection, portfolio: dict, user_email: str) -> None:

    conn.table("users").upsert(
        {"email": user_email, "portfolio": portfolio, 'modified_at': datetime.now().isoformat()},
        on_conflict="email",
    ).execute()
    logger.info('portfolio updated in db')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r


# --- UI Styling ---
st.html("""
<style>
    h1 {
        font-weight: 700 !important;
        color: color-mix(in srgb, var(--text-color) 78%, #16845b);
        letter-spacing: -0.035em;
        padding-bottom: 0.35rem;
    }

    .kpi-card {
        --kpi-accent: #16845b;
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
        border-top: 3px solid var(--kpi-accent);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        min-height: 96px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-card-primary {
        background: color-mix(in srgb, var(--secondary-background-color) 78%, var(--kpi-accent));
        border-color: color-mix(in srgb, var(--text-color) 18%, var(--kpi-accent));
        border-top-color: var(--kpi-accent);
    }

    .kpi-card-income { --kpi-accent: #149766; }
    .kpi-card-target { --kpi-accent: #d39219; }
    .kpi-card-yield { --kpi-accent: #2878c8; }
    .kpi-card-invested { --kpi-accent: #7656b5; }
    .kpi-card-market { --kpi-accent: #168f9c; }

    .kpi-card:not(.kpi-card-primary) {
        background: color-mix(in srgb, var(--secondary-background-color) 92%, var(--kpi-accent));
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: color-mix(in srgb, var(--text-color) 68%, transparent);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: clamp(1.35rem, 2.5vw, 2rem);
        font-weight: 700;
        color: var(--text-color);
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .kpi-delta {
        font-size: 0.875rem;
        font-weight: 500;
    }

    .delta-positive { color: #16a34a; }
    .delta-negative { color: #dc2626; }
    .delta-neutral { color: color-mix(in srgb, var(--text-color) 68%, transparent); }

    @media (max-width: 700px) {
        .kpi-card,
        .kpi-card-primary {
            min-height: 88px;
            padding: 0.7rem 0.8rem;
        }
    }
</style>
""")

def render_kpi(label, value, delta=None, delta_value=None, emphasis=False, tone=None):
    delta_html = ""
    if delta:
        if delta_value is None or delta_value == 0:
            cls = "delta-neutral"
        elif delta_value > 0:
            cls = "delta-positive"
        else:
            cls = "delta-negative"
        delta_html = f'<div class="kpi-delta {cls}">{html.escape(delta)}</div>'

    card_classes = ['kpi-card']
    if emphasis:
        card_classes.append('kpi-card-primary')
    if tone:
        card_classes.append(f'kpi-card-{tone}')
    card_class = ' '.join(card_classes)
    st.html(f"""
        <div class="{card_class}">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            {delta_html}
        </div>
    """)

conn = None
if 'porto_df' not in st.session_state:
    if st.user.is_logged_in:
        try:
            conn = get_db_connection()
            data = get_user_portfolio(conn, st.user.email)
            st.session_state['porto_df'] = normalize_portfolio(data) if data else None
        except Exception:
            logger.exception('Failed to load cloud portfolio')
            st.warning('Your cloud portfolio could not be loaded. You can still enter a portfolio locally.')
            st.session_state['porto_df'] = None
    else:
        st.session_state['porto_df'] = None

# Keep the input panel open until a portfolio exists, regardless of reruns
data_input_expand_flag = st.session_state['porto_df'] is None


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
             if st.button('Sync Portfolio to Cloud', icon=':material/cloud_upload:', width='stretch'):
                 try:
                     conn = conn or get_db_connection()
                     payload = portfolio_to_records(st.session_state['porto_df'])
                     update_user_portfolio(conn, payload, st.user.email)
                     st.success('Portfolio synced successfully.')
                 except Exception:
                     logger.exception('Failed to sync cloud portfolio')
                     st.error('Portfolio sync failed. Your local portfolio is unchanged.')
    else:
         st.button('Log in with Google', icon=':material/login:', on_click=lambda: st.login('google'), width='stretch')


with st.expander('Portfolio Data Input', expanded=data_input_expand_flag):

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

            submit = st.form_submit_button('Load Portfolio Data', icon=':material/upload:', width='stretch')
            
            if submit:
                submitted_df = None
                if method == 'Upload CSV':
                    porto_file = st.session_state.get('porto_file')
                    if porto_file is None or porto_file == 'EMPTY':
                        st.error('Select a CSV file before loading your portfolio.')
                    else:
                        try:
                            porto_file.seek(0)
                            uploaded_df = pd.read_csv(porto_file, sep=',', dtype='str')
                            submitted_df = normalize_portfolio(uploaded_df)
                        except Exception as e:
                            st.error(f'Could not load the CSV: {e}')
                            logger.exception('CSV upload parsing failed')

                elif method == 'Paste Raw':
                    try:
                        submitted_df = parse_raw_portfolio(raw)
                    except Exception as e:
                        st.error(f'Could not parse the pasted portfolio: {e}')
                        logger.exception('Paste Raw parsing failed')

                elif method == 'Form':
                    try:
                        submitted_df = normalize_portfolio(edited_df)
                    except Exception as e:
                        st.error(f'Could not load the form data: {e}')

                if submitted_df is not None:
                    st.session_state['porto_df'] = submitted_df
                    # Toast stays visible even after the input panel collapses
                    st.toast(f'Loaded {len(submitted_df)} portfolio holdings.', icon=':material/check_circle:')
                    logger.info(f'Porto data submitted via {method}: {len(submitted_df)} holdings')


@st.cache_data(max_entries=64, ttl=60*60)
def get_company_profile_data(porto):
    porto = normalize_portfolio(porto)
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

    cp_df['Symbol'] = (
        pd.Index(cp_df.index)
        .astype(str)
        .str.strip()
        .str.upper()
        .str.removesuffix('.JK')
    )
    cp_df = cp_df.drop_duplicates(subset='Symbol', keep='first')

    missing_symbols = porto.loc[~porto['Symbol'].isin(cp_df['Symbol']), 'Symbol'].tolist()
    if missing_symbols:
        market_symbols = [f'{symbol}.JK' for symbol in missing_symbols]
        try:
            live_profiles = hd.get_company_profile(market_symbols)
        except Exception as e:
            logger.warning(
                'Live company profile fallback failed for %s: %s',
                ', '.join(market_symbols),
                e,
            )
        else:
            if live_profiles is not None and not live_profiles.empty:
                live_profiles = live_profiles.copy(deep=True)
                if 'symbol' in live_profiles.columns:
                    live_profiles.set_index('symbol', inplace=True)
                live_profiles['Symbol'] = (
                    pd.Index(live_profiles.index)
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.removesuffix('.JK')
                )
                for col in ['price', 'sector', 'lastDiv']:
                    if col not in live_profiles.columns:
                        live_profiles[col] = np.nan
                # Match the pipeline's definition: latest finalized fiscal-year sum
                live_profiles['lastDiv'] = [
                    get_annual_dividend(symbol, fmp_last_div)
                    for symbol, fmp_last_div in zip(live_profiles['Symbol'], live_profiles['lastDiv'])
                ]
                cp_df = pd.concat(
                    [cp_df, live_profiles[['Symbol', 'price', 'sector', 'lastDiv']]],
                    ignore_index=True,
                ).drop_duplicates(subset='Symbol', keep='first')

    df = porto.merge(
        cp_df[['Symbol', 'price', 'sector', 'lastDiv']],
        on='Symbol',
        how='left',
        validate='one_to_one',
        indicator=True,
    )
    df.rename(columns={'lastDiv': 'div_rate', 'price': 'last_price'}, inplace=True)

    unknown = df.loc[df['_merge'] == 'left_only', 'Symbol'].tolist()
    if len(unknown) == len(df):
        raise ValueError(
            'Market data is unavailable for: '
            f'{", ".join(unknown)}. Remove or correct these symbols before continuing.'
        )

    return df[df['_merge'] == 'both'].drop(columns='_merge').reset_index(drop=True), unknown


def get_annual_dividend(symbol: str, fallback: float) -> float:
    """Return the latest finalized fiscal-year dividend per share, or fallback if history is unavailable."""
    try:
        div_df = hd.get_dividend_history_single_stock_dag(f'{symbol}.JK')
    except Exception as e:
        logger.warning('Dividend history lookup failed for %s: %s', symbol, e)
        return fallback
    required = {'date', 'adjDividend', 'fiscal_year', 'dividend_type'}
    if div_df is None or not required.issubset(div_df.columns):
        return fallback
    return hd.calc_latest_finalized_dividend_sum(div_df)


def validate_market_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop holdings with unusable market fields; return the valid rows and the excluded symbols."""
    validated = df.copy(deep=True)
    rules = {
        'last_price': False,  # must be > 0
        'div_rate': True,     # may be 0
    }
    invalid = pd.Series(False, index=validated.index)
    for column, allow_zero in rules.items():
        validated[column] = pd.to_numeric(validated[column], errors='coerce')
        column_invalid = validated[column].isna() | ~np.isfinite(validated[column])
        column_invalid |= validated[column] < 0 if allow_zero else validated[column] <= 0
        invalid |= column_invalid

    excluded = validated.loc[invalid, 'Symbol'].astype(str).tolist()
    validated = validated[~invalid].reset_index(drop=True)
    if validated.empty:
        raise ValueError(f'Market price or dividend data is invalid for: {", ".join(excluded)}.')

    validated['sector'] = validated['sector'].fillna('Unclassified').replace('', 'Unclassified')
    return validated, excluded


@st.cache_data(max_entries=64, ttl=60*60)
def get_dividend_data(porto):
    stock_list = [x+'.JK' for x in porto['Symbol']]
    divs = {}
    for stock in stock_list:
        try:
            div_df = hd.get_dividend_history_single_stock_dag(stock)
            if div_df is not None:
                if 'dividend_type' in div_df.columns:
                    div_df = div_df[div_df['dividend_type'] != 'special']
                divs[stock] = div_df
            else:
                logger.info(f'stock {stock} does not have dividend history')
        except Exception:
            logger.exception(f'Failed to load dividend history for {stock}')
    return divs


if st.session_state.get('porto_df') is None:
    st.info('Upload or enter your portfolio data above to get started.')
    st.stop()

try:
    st.session_state['porto_df'] = normalize_portfolio(st.session_state['porto_df'])
except ValueError as e:
    st.error(f'Your portfolio could not be loaded: {e}')
    st.stop()

try:
    df, unknown_symbols = get_company_profile_data(st.session_state['porto_df'])
    df, invalid_symbols = validate_market_data(df)
except (ConnectionError, EnvironmentError) as e:
    st.error(f'**Connection error:** {e}')
    logger.exception('Failed to connect to data source')
    st.stop()
except ValueError as e:
    st.error(f'**Data error:** {e}')
    logger.exception('Data error in company profile fetch')
    st.stop()
except KeyError as e:
    st.error(f'**Schema error:** Missing expected column {e}')
    logger.exception('Schema mismatch in company profile data')
    st.stop()
except Exception as e:
    st.error(f'**Unexpected error loading company profile data:** {e}')
    logger.exception('Unexpected error in get_company_profile_data')
    st.stop()

excluded_symbols = unknown_symbols + invalid_symbols
if excluded_symbols:
    st.warning(
        f'Market data is unavailable for **{", ".join(excluded_symbols)}**. '
        'These holdings are excluded from the totals below; check the symbols in Portfolio Data Input.',
        icon=':material/warning:',
    )

try:
    divs = get_dividend_data(st.session_state['porto_df'])
except Exception as e:
    st.warning(f'Could not load dividend history: {e}. Some features may be limited.')
    logger.exception('Failed to load dividend data')
    divs = {}

df['current_lot'] = df['Available Lot'].astype(float)
df['avg_price'] = df['Average Price'].astype(float)

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
summary_cols = st.columns([1.25, 1, 1, 1.15, 1.15])
with summary_cols[0]:
    render_kpi(
        "Annual Dividend Income",
        f"{annual_dividend:,.0f}",
        emphasis=True,
        tone="income",
    )
with summary_cols[1]:
    render_kpi(
        "Income Target Progress",
        f"{achieve_percentage:.2f}%",
        f"Target {target:,.0f}M per year",
        emphasis=True,
        tone="target",
    )
delta_val = total_yield_on_cost - baseline
with summary_cols[2]:
    render_kpi(
        "Yield on Cost",
        f"{total_yield_on_cost:.2f}%",
        f"{delta_val:+.2f}% vs benchmark",
        delta_value=delta_val,
        tone="yield",
    )
with summary_cols[3]:
    render_kpi("Total Invested", f"{total_investment:,.0f}", tone="invested")
market_delta = current_investment_value - total_investment
with summary_cols[4]:
    render_kpi(
        "Market Value",
        f"{current_investment_value:,.0f}",
        f"{market_delta:+,.0f}",
        delta_value=market_delta,
        tone="market",
    )

# Table List
with st.container(border=True):

    # Lazy tabs: the treemap measures its width once on mount, so it must only be
    # created while its tab is visible (a hidden tab has zero width)
    tabs = st.tabs(['Table View', 'Bar Chart View', 'Voronoi Treemap'], key='porto_view_tabs', on_change='rerun')
    
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
                'Dividend / Share',
                format='IDR %,.0f',
                help='Total dividend per share for the latest finalized fiscal year (interim + final)'
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
        div_bar = alt.Chart(df_display).mark_bar(color=cs.CHART_PRIMARY).encode(
            x=alt.X(
                'Symbol:N',
                title='Holding',
                sort=alt.SortField(field='total_dividend', order='descending'),
            ),
            y=alt.Y('total_dividend:Q', title='Annual Dividend (IDR)'),
            tooltip=[
                alt.Tooltip('Symbol:N', title='Holding'),
                alt.Tooltip('total_dividend:Q', title='Annual Dividend', format=',.0f'),
            ],
        )
        yield_bar = alt.Chart(df_display).mark_line(color=cs.CHART_AMBER, point=True).encode(
            x=alt.X(
                'Symbol:N',
                sort=alt.SortField(field='total_dividend', order='descending'),
            ),
            y=alt.Y(
                'yield_on_cost:Q',
                title='Yield on Cost (%)',
                axis=alt.Axis(titleColor=cs.CHART_AMBER),
            ),
            tooltip=[
                alt.Tooltip('Symbol:N', title='Holding'),
                alt.Tooltip('yield_on_cost:Q', title='Yield on Cost', format='.2f'),
            ],
        )
        combined_chart = (div_bar + yield_bar).resolve_scale(y='independent')
        st.altair_chart(combined_chart, width="stretch", theme=None)

    if tabs[2].open:
        with tabs[2]:
            ctrl_cols = st.columns([2, 1])
            value_metric = ctrl_cols[0].selectbox(
                'Value metric',
                options=['total_invested', 'total_dividend'],
                format_func=lambda x: 'Total Invested' if x == 'total_invested' else 'Total Dividend',
                key='vortree_metric'
            )
            treemap_height = ctrl_cols[1].slider('Chart height', 300, 900, 500, key='vortree_height')

            with st.expander('Advanced treemap options', expanded=False):
                ctrl_cols2 = st.columns(4)
                color_scheme = ctrl_cols2[0].selectbox(
                    'Color scheme',
                    ['tableau10', 'category10', 'pastel1'],
                    key='vortree_color'
                )
                show_values = ctrl_cols2[1].checkbox('Show values', value=False, key='vortree_show_values')
                show_pct_only = ctrl_cols2[2].checkbox('Show % only', value=True, key='vortree_pct_only')
                border_color = ctrl_cols2[3].color_picker('Border color', value='#24332c', key='vortree_border_color')
                label_scale = st.number_input(
                    'Label scale', min_value=0.1, max_value=3.0,
                    value=1.5, step=0.1, key='vortree_label_scale'
                )

                if 'vortree_refresh_count' not in st.session_state:
                    st.session_state['vortree_refresh_count'] = 0

                if st.button('Refresh layout', icon=':material/refresh:', key='vortree_refresh'):
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
                border_width=1,
                show_legend=True,
                height=treemap_height,
                key=f'porto_vortree_{st.session_state["vortree_refresh_count"]}'
            )


if main_event.selection.get('rows'):

    symbol = df_display.iloc[main_event.selection['rows'][0]]['Symbol']

    with st.expander('Dividend History', expanded=True):

        if not divs:
            st.warning('Dividend history could not be loaded. Please check your connection.')
        elif symbol+'.JK' not in divs.keys():
            st.info(f'No dividend history available for **{symbol}**.', icon="📭")
        else:
            try:
                div_df = pd.DataFrame(divs[symbol+'.JK'])

                if div_df.empty or 'date' not in div_df.columns or 'adjDividend' not in div_df.columns:
                    st.info(f'Dividend history for **{symbol}** has no payable records.', icon="📭")
                else:
                    div_hist_cols = st.columns([2, 5])
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
                            st.altair_chart(div_bar, width="stretch", theme=None)
                        except Exception as e:
                            st.warning(f'Could not render the dividend history chart: {e}')
                            logger.exception(f'Dividend history chart failed for {symbol}')
            except Exception as e:
                st.error(f'Failed to display dividend history for {symbol}: {e}')
                logger.exception(f'Dividend history display error for {symbol}')

with st.expander('Sector Exposure', expanded=False):

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
            color=alt.Color(
                'sector:N',
                scale=alt.Scale(range=list(cs.CHART_CATEGORY_RANGE)),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip('sector:N', title='Sector'),
                alt.Tooltip('sum(total_dividend):Q', title='Annual Dividend', format=',d'),
            ]
        ).properties(height=250)
        st.altair_chart(sector_pie, width="stretch", theme=None)


with st.expander('Dividend Timeline', expanded=False):

    view_cols = st.columns([2, 1])
    with view_cols[0]:
        view_type = st.segmented_control('View Pattern', ['Calendar', 'Monthly Bar', 'Grid Table'], default='Monthly Bar')

    div_lists = build_last_year_dividends(df, divs)

    if not div_lists:
        st.info('No dividend payment data was found in the past 12 months.')
    else:
        render_dividend_timeline(div_lists, view_type)

# Project future earnings
with st.expander('Compounding Projection', expanded=False):
    st.markdown("Estimate your future returns based on compounding dividends and yield growth.")

    proj_input_cols = st.columns([1, 1, 3])
    with proj_input_cols[0]:
        number_of_year = st.number_input('Years', value=25, min_value=1, max_value=50)
    with proj_input_cols[1]:
        current_yield = annual_dividend / current_investment_value * 100
        inc = st.number_input(
            'Expected Annual Income Growth (%)',
            value=round(float(np.clip(current_yield, 0.0, PROJECTION_MAX_GROWTH)), 2),
            min_value=0.0,
            max_value=PROJECTION_MAX_GROWTH,
            step=0.1,
            help=(
                'Defaults to your current yield at market price: reinvesting dividends at '
                "today's prices grows income by roughly this rate. Add expected company "
                'dividend growth on top if you want to include it.'
            ),
        )

    df_future = project_income(annual_dividend, total_investment, inc, number_of_year)
    df_future['achieved'] = df_future['returns'] > (target*1_000_000)

    base_chart = alt.Chart(df_future)
    return_chart = base_chart.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('years:N', title='Compounding Journey'),
        y=alt.Y('returns:Q', title='Annual Income (IDR)'),
        color=alt.condition(
            alt.datum['achieved'],
            alt.value(cs.CHART_POSITIVE),
            alt.value(cs.CHART_BLUE),
        ),
        tooltip=['years', alt.Tooltip('returns', format=',.0f')]
    )

    yield_chart = base_chart.mark_line(point=True, color=cs.CHART_AMBER).encode(
        x=alt.X('years:N'),
        y=alt.Y('yield:Q', title='Yield on Total Cost (%)'),
        tooltip=['years', alt.Tooltip('yield', title='Yield on cost incl. reinvested', format='.2f')]
    )

    future_chart = (return_chart + yield_chart).resolve_scale(y='independent').properties(height=400)
    st.altair_chart(future_chart, width="stretch", theme=None)
