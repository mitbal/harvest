import os
import html
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
import harvest.portfolio as hpf
from harvest import chart_style as cs
from harvest.utils import setup_logging

# Must be the very first Streamlit command
st.set_page_config(page_title='Portfolio Analytics - Panen Dividen', layout='wide')
hp.enable_chart_theme()

PROJECTION_MAX_GROWTH = 50.0  # upper bound for the income growth input (%)
DIVIDEND_FETCH_WORKERS = 8


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
    elif view_type == 'Monthly bar':
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
        for first_month in (1, 5, 9):
            month_cols = st.columns(4)
            for column, month in zip(month_cols, range(first_month, first_month + 4)):
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


def run_db(operation):
    """Run ``operation(conn)``; if the cached connection fails (e.g. expired session), reconnect once."""
    try:
        return operation(get_db_connection())
    except Exception:
        logger.warning('Supabase call failed; reconnecting and retrying once', exc_info=True)
        get_db_connection.clear()
        return operation(get_db_connection())


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
        font-size: clamp(1.3rem, 2.1vw, 1.85rem);
        white-space: nowrap;
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

if 'porto_df' not in st.session_state:
    if st.user.is_logged_in:
        try:
            data = run_db(lambda conn: get_user_portfolio(conn, st.user.email))
            st.session_state['porto_df'] = hpf.normalize_portfolio(data) if data else None
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
        st.button('Log out', icon=':material/logout:', on_click=st.logout, width='stretch')
        if st.session_state.get('porto_df') is not None:
            if st.button('Sync portfolio to cloud', icon=':material/cloud_upload:', width='stretch'):
                try:
                    payload = hpf.portfolio_to_records(st.session_state['porto_df'])
                    run_db(lambda conn: update_user_portfolio(conn, payload, st.user.email))
                    st.toast('Portfolio synced to the cloud.', icon=':material/cloud_done:')
                except Exception:
                    logger.exception('Failed to sync cloud portfolio')
                    st.error('Portfolio sync failed. Your local portfolio is unchanged.')
    else:
        st.button('Log in with Google', icon=':material/login:', on_click=lambda: st.login('google'), width='stretch')


with st.expander('Portfolio data', icon=':material/edit_note:', expanded=data_input_expand_flag):

    method = st.segmented_control(
        'Import method',
        ['Form', 'Upload CSV', 'Paste from Stockbit'],
        default='Form',
        required=True,
        key='porto_import_method',
    )
    method_help = {
        'Form': 'Edit holdings directly. Lots are in units of 100 shares; average price is per share.',
        'Upload CSV': 'CSV with columns Symbol, Available Lot and Average Price.',
        'Paste from Stockbit': 'Copy the holdings table from the Stockbit portfolio page and paste it here.',
    }
    st.caption(method_help[method])

    with st.form('portfolio_input_form', border=False):
        if method == 'Upload CSV':
            uploaded_file = st.file_uploader('Select your portfolio CSV', type='csv')
            if uploaded_file:
                st.session_state['porto_file'] = uploaded_file
            
        elif method == 'Paste from Stockbit':
            raw = st.text_area('Stockbit holdings table', height=200, placeholder='BBCA 10 ... 9,500 ...')
        
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

        submit = st.form_submit_button('Load portfolio', icon=':material/upload:', type='primary', width='stretch')
        
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
                        submitted_df = hpf.normalize_portfolio(uploaded_df)
                    except Exception as e:
                        st.error(f'Could not load the CSV: {e}')
                        logger.exception('CSV upload parsing failed')

            elif method == 'Paste from Stockbit':
                try:
                    submitted_df = hpf.parse_raw_portfolio(raw)
                except Exception as e:
                    st.error(f'Could not parse the pasted portfolio: {e}')
                    logger.exception('Paste Raw parsing failed')

            elif method == 'Form':
                try:
                    submitted_df = hpf.normalize_portfolio(edited_df)
                except Exception as e:
                    st.error(f'Could not load the form data: {e}')

            if submitted_df is not None:
                st.session_state['porto_df'] = submitted_df
                # Toast stays visible even after the input panel collapses
                st.toast(f'Loaded {len(submitted_df)} portfolio holdings.', icon=':material/check_circle:')
                logger.info(f'Porto data submitted via {method}: {len(submitted_df)} holdings')
                # Rerun so the input panel collapses and the dashboard comes into view
                st.rerun()


def fetch_dividend_history(stock: str):
    """Return a stock's dividend history, or None when it has none or the lookup fails."""
    try:
        return hd.get_dividend_history_single_stock_dag(stock)
    except Exception as e:
        logger.warning('Dividend history lookup failed for %s: %s', stock, e)
        return None


def fetch_live_profiles(symbols: list[str]) -> pd.DataFrame | None:
    """Fetch FMP profiles for symbols missing from the cache, with the pipeline's dividend definition."""
    market_symbols = [f'{symbol}.JK' for symbol in symbols]
    try:
        live_profiles = hd.get_company_profile(market_symbols)
    except Exception as e:
        logger.warning('Live company profile fallback failed for %s: %s', ', '.join(market_symbols), e)
        return None
    if live_profiles is None or live_profiles.empty:
        return None

    live_profiles = hpf.with_symbol_column(live_profiles)
    # Match the pipeline's definition: latest finalized fiscal-year sum
    with ThreadPoolExecutor(max_workers=DIVIDEND_FETCH_WORKERS) as pool:
        histories = list(pool.map(fetch_dividend_history, [f'{s}.JK' for s in live_profiles['Symbol']]))
    live_profiles['lastDiv'] = [
        hpf.annual_dividend_from_history(history, fmp_last_div)
        for history, fmp_last_div in zip(histories, live_profiles['lastDiv'])
    ]
    return live_profiles


@st.cache_data(max_entries=64, ttl=60*60)
def get_company_profile_data(porto):
    porto = hpf.normalize_portfolio(porto)
    redis_url = os.environ.get('REDIS_URL')
    if not redis_url:
        raise EnvironmentError('REDIS_URL environment variable is not set.')

    try:
        r = connect_redis(redis_url)
        raw = r.get('div_score_jkse')
    except Exception as e:
        raise ConnectionError(f'Failed to connect to Redis or fetch data: {e}') from e

    if raw is None:
        raise ValueError('No data found in Redis for key "div_score_jkse". Please ensure the data pipeline has run.')

    profiles = hpf.parse_company_profiles(raw)

    missing_symbols = porto.loc[~porto['Symbol'].isin(profiles['Symbol']), 'Symbol'].tolist()
    if missing_symbols:
        live_profiles = fetch_live_profiles(missing_symbols)
        if live_profiles is not None:
            profiles = pd.concat(
                [profiles, live_profiles[['Symbol'] + hpf.PROFILE_COLUMNS]],
                ignore_index=True,
            ).drop_duplicates(subset='Symbol', keep='first')

    return hpf.merge_market_data(porto, profiles)


@st.cache_data(max_entries=64, ttl=60*60)
def get_dividend_data(porto):
    """Fetch dividend histories for all holdings in parallel, excluding special dividends."""
    stock_list = [x + '.JK' for x in porto['Symbol']]
    with ThreadPoolExecutor(max_workers=DIVIDEND_FETCH_WORKERS) as pool:
        histories = pool.map(fetch_dividend_history, stock_list)

    divs = {}
    for stock, div_df in zip(stock_list, histories):
        if div_df is None:
            logger.info(f'stock {stock} does not have dividend history')
            continue
        if 'dividend_type' in div_df.columns:
            div_df = div_df[div_df['dividend_type'] != 'special']
        divs[stock] = div_df
    return divs


if st.session_state.get('porto_df') is None:
    st.info('Upload or enter your portfolio data above to get started.')
    st.stop()

try:
    st.session_state['porto_df'] = hpf.normalize_portfolio(st.session_state['porto_df'])
except ValueError as e:
    st.error(f'Your portfolio could not be loaded: {e}')
    st.stop()

try:
    df, unknown_symbols = get_company_profile_data(st.session_state['porto_df'])
    df, invalid_symbols = hpf.validate_market_data(df)
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
        'These holdings are excluded from the totals below; check the symbols in Portfolio data.',
        icon=':material/warning:',
    )

try:
    divs = get_dividend_data(st.session_state['porto_df'])
except Exception as e:
    st.warning(f'Could not load dividend history: {e}. Some features may be limited.')
    logger.exception('Failed to load dividend data')
    divs = {}

# --- Goals ---
with st.container(horizontal=True, horizontal_alignment='distribute', vertical_alignment='center'):
    st.caption(':material/update: Market prices and dividends come from the daily data pipeline.')
    with st.popover('Income goal', icon=':material/flag:'):
        target = st.number_input(
            'Target annual income (IDR million)',
            value=240, step=1, min_value=1, max_value=10_000,
            format='%d',
            key='porto_target',
            help='The yearly dividend income you are working towards',
        )
        baseline = st.number_input(
            'Benchmark yield (%)',
            value=6.35, step=0.01, min_value=0.01, max_value=99.99,
            key='porto_benchmark',
            help='Yield to compare against, e.g. a government bond or deposit rate',
        )

df = hpf.enrich_holdings(df)
summary = hpf.summarize_portfolio(df, target)
annual_dividend = summary['annual_dividend']
total_investment = summary['total_invested']
total_yield_on_cost = summary['yield_on_cost']

df_display = df[['Symbol', 'Available Lot', 'avg_price', 'total_invested', 'div_rate', 'last_price',
                 'yield_on_cost', 'yield_on_price', 'total_dividend']].copy(deep=True)


# --- KPIs ---
summary_cols = st.columns([1.25, 1, 1, 1.15, 1.15])
with summary_cols[0]:
    render_kpi(
        "Annual dividend income",
        hpf.format_idr(annual_dividend),
        f"{hpf.format_idr(annual_dividend / 12)} per month",
        emphasis=True,
        tone="income",
    )
with summary_cols[1]:
    render_kpi(
        "Goal progress",
        f"{summary['target_progress']:.1f}%",
        f"of {hpf.format_idr(target * 1_000_000)} per year",
        emphasis=True,
        tone="target",
    )
delta_val = total_yield_on_cost - baseline
with summary_cols[2]:
    render_kpi(
        "Yield on cost",
        f"{total_yield_on_cost:.2f}%",
        f"{delta_val:+.2f} pts vs {baseline:.2f}% benchmark",
        delta_value=delta_val,
        tone="yield",
    )
with summary_cols[3]:
    render_kpi(
        "Total invested",
        hpf.format_idr(total_investment),
        f"{len(df)} holdings · {df['sector'].nunique()} sectors",
        tone="invested",
    )
market_delta = summary['market_delta']
market_return = market_delta / total_investment * 100
with summary_cols[4]:
    render_kpi(
        "Market value",
        hpf.format_idr(summary['market_value']),
        f"{hpf.format_idr(market_delta, signed=True)} ({market_return:+.2f}%)",
        delta_value=market_delta,
        tone="market",
    )


# --- Holdings ---
with st.container(border=True):

    # Lazy tabs: only compute the treemap layout while its tab is open
    tabs = st.tabs(
        [':material/table: Holdings', ':material/bar_chart: Income by holding', ':material/bubble_chart: Allocation map'],
        key='porto_view_tabs',
        on_change='rerun',
    )

    with tabs[0]:
        st.caption(':material/touch_app: Select a row to see that holding\'s dividend history and 5-year outlook.')

        cfig = {
            'Symbol': st.column_config.TextColumn('Symbol', pinned=True),
            'Available Lot': st.column_config.NumberColumn('Lots', format='%,d'),
            'avg_price': st.column_config.NumberColumn('Avg price', format='%,d'),
            'last_price': st.column_config.NumberColumn('Market price', format='%,d'),
            'total_invested': st.column_config.NumberColumn('Invested (IDR)', format='%,d'),
            'div_rate': st.column_config.NumberColumn(
                'Dividend / share',
                format='%,.0f',
                help='Total dividend per share for the latest finalized fiscal year (interim + final)'
            ),
            'yield_on_cost': st.column_config.NumberColumn(
                'Yield on cost',
                format='%.2f%%',
                help='Dividend yield based on your average purchase price'
            ),
            'yield_on_price': st.column_config.NumberColumn(
                'Yield on price',
                format='%.2f%%',
                help='Dividend yield based on the current market price'
            ),
            'total_dividend': st.column_config.ProgressColumn(
                'Annual dividend (IDR)',
                format='%,d',
                min_value=0,
                max_value=int(df_display['total_dividend'].max()) or 1,
                help='Bar length is relative to your largest income holding'
            ),
        }

        main_event = st.dataframe(
            df_display,
            on_select='rerun',
            selection_mode='single-row',
            column_config=cfig,
            hide_index=True,
            width='stretch',
            key='porto_holdings_table',
        )

    with tabs[1]:
        income_df = df_display.assign(
            share=lambda d: d['total_dividend'] / d['total_dividend'].sum() * 100 if d['total_dividend'].sum() else 0
        ).sort_values('total_dividend', ascending=False)
        income_base = alt.Chart(income_df).encode(
            y=alt.Y('Symbol:N', sort=income_df['Symbol'].tolist(), title=None),
            x=alt.X(
                'total_dividend:Q', title='Annual dividend (IDR)',
                axis=alt.Axis(format='~s', tickCount=5),
                # Headroom for the share labels at the end of the longest bar
                scale=alt.Scale(domain=[0, max(income_df['total_dividend'].max(), 1) * 1.12]),
            ),
        )
        income_bar = income_base.mark_bar(
            color=cs.CHART_PRIMARY, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height={'band': 0.7},
        ).encode(
            tooltip=[
                alt.Tooltip('Symbol:N', title='Holding'),
                alt.Tooltip('total_dividend:Q', title='Annual dividend (IDR)', format=',.0f'),
                alt.Tooltip('share:Q', title='Share of income (%)', format='.1f'),
                alt.Tooltip('yield_on_cost:Q', title='Yield on cost (%)', format='.2f'),
                alt.Tooltip('yield_on_price:Q', title='Yield on price (%)', format='.2f'),
            ],
        )
        income_labels = income_base.mark_text(align='left', dx=4, color=cs.CHART_MUTED_TEXT, fontSize=11).transform_calculate(
            label="format(datum.share, '.1f') + '%'"
        ).encode(text='label:N')
        st.altair_chart(
            (income_bar + income_labels).properties(height=max(240, 26 * len(income_df))),
            width='stretch',
            theme=None,
        )
        st.caption('Labels show each holding\'s share of annual income. Hover a bar for its yields.')

    if tabs[2].open:
        with tabs[2]:
            value_metric = st.segmented_control(
                'Size by',
                options=['total_invested', 'total_dividend'],
                format_func=lambda x: 'Amount invested' if x == 'total_invested' else 'Annual dividend',
                default='total_invested',
                required=True,
                key='vortree_metric',
            )
            treemap_df = df_display[['Symbol', value_metric]].copy()
            treemap_df['sector'] = df['sector'].values
            st_vortree(
                treemap_df,
                name_col='Symbol',
                value_col=value_metric,
                group_col='sector',
                color_scheme='tableau10',
                show_pct_only=True,
                label_scale=1.5,
                border_color='#24332c',
                border_width=1,
                show_legend=True,
                height=520,
                key='porto_vortree',
            )
            st.caption('Cells are grouped by sector; area shows each holding\'s share.')


if main_event.selection.get('rows'):

    selected = df_display.iloc[main_event.selection['rows'][0]]
    symbol = selected['Symbol']

    with st.expander(f'Dividend history · {symbol}', icon=':material/history:', expanded=True):

        if not divs:
            st.warning('Dividend history could not be loaded. Please check your connection.')
        elif symbol+'.JK' not in divs.keys():
            st.info(f'No dividend history available for **{symbol}**.', icon=':material/event_busy:')
        else:
            try:
                div_df = pd.DataFrame(divs[symbol+'.JK'])

                if div_df.empty or 'date' not in div_df.columns or 'adjDividend' not in div_df.columns:
                    st.info(f'Dividend history for **{symbol}** has no payable records.', icon=':material/event_busy:')
                else:
                    div_hist_cols = st.columns([2, 5])
                    with div_hist_cols[0]:
                        st.dataframe(
                            div_df[['date', 'adjDividend']],
                            column_config={
                                'date': st.column_config.DateColumn('Ex-date'),
                                'adjDividend': st.column_config.NumberColumn('Dividend / share', format='%,.1f')
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
                                last_val=selected['div_rate'],
                                inc_val=stats['historical_mean_flat']
                            )
                            st.altair_chart(div_bar, width="stretch", theme=None)
                        except Exception as e:
                            st.warning(f'Could not render the dividend history chart: {e}')
                            logger.exception(f'Dividend history chart failed for {symbol}')
            except Exception as e:
                st.error(f'Failed to display dividend history for {symbol}: {e}')
                logger.exception(f'Dividend history display error for {symbol}')

with st.expander('Sector exposure', icon=':material/donut_small:', expanded=False):

    sector_df = (
        df.groupby('sector')['total_dividend'].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total_income = sector_df['total_dividend'].sum()
    sector_df['share'] = sector_df['total_dividend'] / total_income * 100 if total_income else 0.0
    sector_df['holdings'] = sector_df['sector'].map(df['sector'].value_counts())

    sector_cols = st.columns([3, 2])
    with sector_cols[0]:
        st.markdown('**Dividend by sector**')
        st.caption(
            f'{len(sector_df)} sectors · IDR {total_income:,.0f} a year. '
            'Select a row to see its holdings.'
        )
        sector_event = st.dataframe(
            sector_df[['sector', 'holdings', 'total_dividend', 'share']],
            on_select='rerun',
            selection_mode='single-row',
            hide_index=True,
            width='stretch',
            key='porto_sector_table',
            column_config={
                'sector': st.column_config.TextColumn('Sector'),
                'holdings': st.column_config.NumberColumn('Stocks', format='%d', width='small'),
                'total_dividend': st.column_config.NumberColumn('Dividend (IDR)', format='%,d'),
                'share': st.column_config.ProgressColumn(
                    'Share',
                    format='%.1f%%',
                    min_value=0,
                    max_value=float(sector_df['share'].max()) or 1.0,
                    help='Bar length is relative to your largest income sector',
                ),
            },
        )

    with sector_cols[1]:
        selected_rows = sector_event.selection.rows
        if not selected_rows:
            st.markdown('**Income mix**')
            st.caption('Share of your annual dividend from each sector.')
            sector_colors = [
                cs.CHART_CATEGORY_RANGE[i % len(cs.CHART_CATEGORY_RANGE)] for i in range(len(sector_df))
            ]
            donut_base = alt.Chart(sector_df).encode(
                theta=alt.Theta('total_dividend:Q', stack=True),
                order=alt.Order('total_dividend:Q', sort='descending'),
            )
            # Radii follow the plot size so the ring never outgrows a narrow column
            donut = donut_base.mark_arc(
                innerRadius=alt.ExprRef('min(width, height) * 0.3'),
                outerRadius=alt.ExprRef('min(width, height) * 0.5 - 4'),
                stroke='white',
                strokeWidth=2,
            ).encode(
                color=alt.Color(
                    'sector:N',
                    scale=alt.Scale(domain=sector_df['sector'].tolist(), range=sector_colors),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip('sector:N', title='Sector'),
                    alt.Tooltip('total_dividend:Q', title='Annual dividend (IDR)', format=',.0f'),
                    alt.Tooltip('share:Q', title='Share of income (%)', format='.1f'),
                ],
            )
            center_base = alt.Chart(pd.DataFrame({'total': [f'IDR {total_income / 1e6:,.1f}M'], 'unit': ['per year']}))
            center = (
                center_base.mark_text(dy=-8, fontSize=15, fontWeight='bold', color=cs.CHART_TEXT).encode(text='total:N')
                + center_base.mark_text(dy=11, fontSize=11, color=cs.CHART_MUTED_TEXT).encode(text='unit:N')
            )
            st.altair_chart(
                (donut + center).properties(height=240).configure_view(stroke=None),
                width='stretch',
                theme=None,
            )
            # HTML legend wraps on narrow columns instead of clipping like a Vega legend
            legend_items = ''.join(
                f'<span style="display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;'
                f'font-size:0.8rem;color:{cs.CHART_TEXT};white-space:nowrap">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{color};flex:none"></span>'
                f'{html.escape(str(name))} <span style="color:{cs.CHART_MUTED_TEXT}">{share:.1f}%</span></span>'
                for name, share, color in zip(sector_df['sector'], sector_df['share'], sector_colors)
            )
            st.markdown(f'<div style="display:flex;flex-wrap:wrap">{legend_items}</div>', unsafe_allow_html=True)
        else:
            sector_name = sector_df['sector'].iloc[selected_rows[0]]
            sector_row = sector_df.iloc[selected_rows[0]]
            st.markdown(f'**{sector_name} holdings**')
            st.caption(
                f'{int(sector_row["holdings"])} stock{"s" if sector_row["holdings"] != 1 else ""} · '
                f'{sector_row["share"]:.1f}% of your dividend income. Clear the selection to see the mix.'
            )
            st.dataframe(
                df[df['sector'] == sector_name][['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False),
                hide_index=True,
                width='stretch',
                column_config={
                    'total_dividend': st.column_config.NumberColumn('Dividend (IDR)', format='%,d'),
                }
            )

with st.expander('Dividend timeline', icon=':material/calendar_month:', expanded=False):

    view_type = st.segmented_control(
        'View',
        ['Monthly bar', 'Calendar', 'Grid table'],
        default='Monthly bar',
        required=True,
        key='porto_timeline_view',
    )
    st.caption('Payments received in the last 12 months, at your current lot sizes.')

    div_lists = hpf.build_last_year_dividends(df, divs)

    if not div_lists:
        st.info('No dividend payment data was found in the past 12 months.')
    else:
        render_dividend_timeline(div_lists, view_type)

# Project future earnings
with st.expander('Compounding projection', icon=':material/trending_up:', expanded=False):
    st.caption('Estimate future income if every dividend is reinvested and income grows at a steady rate.')

    proj_input_cols = st.columns([1, 1, 3])
    with proj_input_cols[0]:
        number_of_year = st.number_input('Years', value=25, min_value=1, max_value=50)
    with proj_input_cols[1]:
        inc = st.number_input(
            'Expected annual income growth (%)',
            value=round(float(np.clip(summary['yield_on_price'], 0.0, PROJECTION_MAX_GROWTH)), 2),
            min_value=0.0,
            max_value=PROJECTION_MAX_GROWTH,
            step=0.1,
            help=(
                'Defaults to your current yield at market price: reinvesting dividends at '
                "today's prices grows income by roughly this rate. Add expected company "
                'dividend growth on top if you want to include it.'
            ),
        )

    df_future = hpf.project_income(annual_dividend, total_investment, inc, number_of_year)
    target_idr = target * 1_000_000
    reached = df_future[df_future['returns'] >= target_idr]
    if annual_dividend >= target_idr:
        st.markdown(f':material/check_circle: You already earn your goal of **{hpf.format_idr(target_idr)}** a year.')
    elif not reached.empty:
        years_to_goal = int(reached.index[0]) + 1
        st.markdown(
            f':material/flag: At this rate you reach **{hpf.format_idr(target_idr)}** a year in '
            f'**{years_to_goal} year{"s" if years_to_goal > 1 else ""}**.'
        )
    else:
        st.markdown(f':material/flag: At this rate you don\'t reach **{hpf.format_idr(target_idr)}** within {number_of_year} years.')

    base_chart = alt.Chart(df_future)
    return_chart = base_chart.mark_bar(
        color=cs.CHART_PRIMARY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
    ).encode(
        x=alt.X('years:N', title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('returns:Q', title='Annual income (IDR)', axis=alt.Axis(format='~s')),
        tooltip=[
            alt.Tooltip('years:N', title='Year'),
            alt.Tooltip('returns:Q', title='Annual income (IDR)', format=',.0f'),
            alt.Tooltip('yield:Q', title='Yield on total cost (%)', format='.2f'),
        ],
    )
    target_rule = alt.Chart(pd.DataFrame({'target': [target_idr]})).mark_rule(
        color=cs.CHART_AMBER, strokeDash=[6, 4], strokeWidth=2,
    ).encode(y='target:Q')
    target_label = alt.Chart(pd.DataFrame({'target': [target_idr], 'label': ['Income goal']})).mark_text(
        align='left', baseline='bottom', dx=4, dy=-4, color=cs.CHART_TEXT, fontSize=11,
    ).encode(y='target:Q', x=alt.value(0), text='label:N')

    future_chart = (return_chart + target_rule + target_label).properties(height=400)
    st.altair_chart(future_chart, width="stretch", theme=None)
