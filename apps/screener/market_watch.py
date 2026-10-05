import os
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import redis
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts5 import st_echarts

import harvest.plot as hp
import harvest.data as hd
import harvest.prices as hprices
from harvest import chart_style as cs
from harvest.utils import setup_logging
from harvest.viewport import remaining_viewport_height


st.set_page_config(page_title='Market Heatmap - Panen Dividen', layout='wide')
hp.enable_chart_theme()

# The heatmap should fill the first screen, so this page trims Streamlit's
# default top padding and uses a compact title. Scoped to this page's elements.
st.html("""
<style>
    [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
    .st-key-mw_header h1 { font-size: 1.75rem; padding: 0; }
</style>
""")
with st.container(key='mw_header'):
    st.title(
        'Market Heatmap',
        help=(
            'Tile size and colour follow the selected measures. '
            'Click a stock tile to open its full analysis in the Ranking Screener; '
            'click a sector to zoom in.'
        ),
    )
toolbar = st.container(horizontal=True, vertical_alignment='bottom', key='mw_toolbar')
stats_slot = st.container()

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


logger = get_logger('market_watch')


# ── Load stock universe (sector / industry / mcap metadata) ──────────────── #

def parse_live_profile_item(item: dict) -> dict | None:
    """Normalize one FMP profile quote without letting a bad row abort its chunk."""
    try:
        price = item.get('price')
        changes = item.get('changes')
        changes_pct = item.get('changesPercentage')
        symbol = item.get('symbol')
        if price is None or symbol is None:
            return None

        price = float(price)
        if changes is not None:
            prev_close = price - float(changes)
            ret = (
                float(changes_pct)
                if changes_pct is not None
                else (price / prev_close - 1) * 100 if prev_close else np.nan
            )
        elif changes_pct is not None:
            ret = float(changes_pct)
            denominator = 1 + ret / 100
            prev_close = price / denominator if denominator else np.nan
        else:
            return None

        if not np.isfinite([price, prev_close, ret]).all():
            return None
        return {
            'symbol': symbol,
            'close': price,
            'prev_close': prev_close,
            'return_1d_pct': ret,
        }
    except (TypeError, ValueError, OverflowError):
        return None

@st.cache_data(max_entries=32, ttl=60 * 5, show_spinner='Fetching live price changes from FMP…')
def get_live_returns_from_profile(symbols: tuple) -> pd.DataFrame:
    """
    Fetches the latest price, absolute change, and % change for each symbol
    from the FMP ``/profile`` endpoint.  This is used as a real-time fallback
    when today's stored prices are not yet available.

    Returns a DataFrame indexed by symbol with columns:
        [close, prev_close, return_1d_pct]
    matching the shape produced by ``calc_daily_return_for_date``.
    """
    import requests as _req

    chunk_size = 50
    chunks = [list(symbols[i:i + chunk_size]) for i in range(0, len(symbols), chunk_size)]

    def fetch_chunk(chunk):
        rows = []
        param = ','.join(chunk)
        url = (
            f'https://financialmodelingprep.com/api/v3/profile/{param}'
            f'?apikey={api_key}'
        )
        try:
            resp = _req.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for item in data:
                    parsed = parse_live_profile_item(item)
                    if parsed is not None:
                        rows.append(parsed)
        except Exception as e:
            logger.warning(f'Could not fetch company profile chunk: {e}')
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=min(4, len(chunks) or 1)) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            rows.extend(future.result())

    if not rows:
        return pd.DataFrame(columns=['close', 'prev_close', 'return_1d_pct'])

    df = pd.DataFrame(rows).set_index('symbol')
    return df


@st.cache_data(max_entries=4, ttl=60 * 10, show_spinner='Loading stock universe…')
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


# ── Load historical prices for a date range ──────────────────────────────── #

@st.cache_data(max_entries=16, ttl=60 * 60, show_spinner='Fetching historical prices…')
def get_prices_for_date_range(
    exchange: str,
    symbols: tuple,       # tuple so it is hashable for caching
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """
    Close prices for ``symbols`` between ``date_from`` and ``date_to`` (inclusive)
    from the Parquet price history. Returns a DataFrame with columns [symbol, date, close].
    """
    try:
        return hprices.load_prices(exchange, symbols=symbols, start=date_from, end=date_to)
    except Exception as e:
        logger.error(f'Price history error: {e}')
        return pd.DataFrame(columns=['symbol', 'date', 'close'])


# ── Fetch index prices from FMP ───────────────────────────────────────────── #

_INDEX_SYMBOLS = {
    '^JKSE':  'IHSG',
    '^GSPC':  'S&P 500',
    '^N225':  'Nikkei 225',
    '^KS11':  'KOSPI',
    '^AXJO':  'ASX 200',
    '^TWII':  'TWSE',
    '^HSI':   'Hang Seng',
    '^STI':   'STI',
    '^KLSE':  'KLCI',
    '^BSESN': 'Sensex',
}

# IHSG is the anchor line, drawn thick in ink; the rest keep one hue each so a
# line never changes colour when the selection changes.
_INDEX_COLORS = {
    'IHSG': cs.CHART_TEXT,
    **dict(zip(
        ['S&P 500', 'Nikkei 225', 'KOSPI', 'ASX 200', 'TWSE',
         'Hang Seng', 'STI', 'KLCI', 'Sensex'],
        cs.CHART_COMPARISON_RANGE,
    )),
}

_INDEX_FLAGS = {
    'IHSG':       '🇮🇩',
    'S&P 500':    '🇺🇸',
    'Nikkei 225': '🇯🇵',
    'KOSPI':      '🇰🇷',
    'ASX 200':    '🇦🇺',
    'TWSE':       '🇹🇼',
    'Hang Seng':  '🇭🇰',
    'STI':        '🇸🇬',
    'KLCI':       '🇲🇾',
    'Sensex':     '🇮🇳',
}


@st.cache_data(max_entries=32, ttl=60 * 60 * 4, show_spinner='Fetching index prices…')
def get_index_prices(
    fmp_key: str,
    date_from: str,
    date_to: str,
    symbols: tuple | None = None,
) -> pd.DataFrame:
    """
    Fetches daily close prices for all tracked indices from FMP.
    Returns a long-format DataFrame: [date, index, close].
    """
    import requests as _req

    all_rows = []
    selected_symbols = tuple(_INDEX_SYMBOLS) if symbols is None else symbols
    for sym in selected_symbols:
        label = _INDEX_SYMBOLS[sym]
        try:
            url = (
                f'https://financialmodelingprep.com/api/v3/historical-price-full/{sym}'
                f'?from={date_from}&to={date_to}&apikey={fmp_key}'
            )
            resp = _req.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            hist = data.get('historical', [])
            for row in hist:
                all_rows.append({
                    'date':  row['date'],
                    'index': label,
                    'close': row['close'],
                })
        except Exception as e:
            logger.warning(f'Could not fetch index {sym} ({label}): {e}')

    if not all_rows:
        return pd.DataFrame(columns=['date', 'index', 'close'])

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


# ── Currency (Forex vs IDR) definitions ──────────────────────────────────── #

# FMP forex pair symbols: {pair_symbol: (label, flag, color)}
_FX_SYMBOLS = {
    'USDIDR': ('USD/IDR', '🇺🇸', cs.CHART_COMPARISON_RANGE[0]),
    'EURIDR': ('EUR/IDR', '🇪🇺', cs.CHART_COMPARISON_RANGE[1]),
    'SGDIDR': ('SGD/IDR', '🇸🇬', cs.CHART_COMPARISON_RANGE[2]),
    'JPYIDR': ('JPY/IDR', '🇯🇵', cs.CHART_COMPARISON_RANGE[3]),
    'GBPIDR': ('GBP/IDR', '🇬🇧', cs.CHART_COMPARISON_RANGE[4]),
    'AUDIDR': ('AUD/IDR', '🇦🇺', cs.CHART_COMPARISON_RANGE[5]),
    'CNYIDR': ('CNY/IDR', '🇨🇳', cs.CHART_COMPARISON_RANGE[6]),
    'HKDIDR': ('HKD/IDR', '🇭🇰', cs.CHART_COMPARISON_RANGE[7]),
    'MYRIDR': ('MYR/IDR', '🇲🇾', cs.CHART_COMPARISON_RANGE[8]),
    'KRWIDR': ('KRW/IDR', '🇰🇷', cs.CHART_COMPARISON_RANGE[9]),
}

_FX_LABELS  = {sym: v[0] for sym, v in _FX_SYMBOLS.items()}
_FX_FLAGS   = {sym: v[1] for sym, v in _FX_SYMBOLS.items()}
_FX_COLORS  = {sym: v[2] for sym, v in _FX_SYMBOLS.items()}

# Label → symbol reverse map (for lookups)
_FX_LABEL_TO_SYM = {v[0]: sym for sym, v in _FX_SYMBOLS.items()}


@st.cache_data(max_entries=32, ttl=60 * 60 * 4, show_spinner='Fetching currency rates…')
def get_fx_prices(
    fmp_key: str,
    date_from: str,
    date_to: str,
    symbols: tuple | None = None,
) -> pd.DataFrame:
    """
    Fetches daily close rates for all tracked FX pairs (vs IDR) from FMP.
    Returns a long-format DataFrame: [date, pair, close].
    """
    import requests as _req

    all_rows = []
    selected_symbols = tuple(_FX_SYMBOLS) if symbols is None else symbols
    for sym in selected_symbols:
        label, _flag, _color = _FX_SYMBOLS[sym]
        try:
            url = (
                f'https://financialmodelingprep.com/api/v3/historical-price-full/{sym}'
                f'?from={date_from}&to={date_to}&apikey={fmp_key}'
            )
            resp = _req.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            hist = data.get('historical', [])
            for row in hist:
                all_rows.append({
                    'date':  row['date'],
                    'pair':  label,
                    'close': row['close'],
                })
        except Exception as e:
            logger.warning(f'Could not fetch FX pair {sym} ({label}): {e}')

    if not all_rows:
        return pd.DataFrame(columns=['date', 'pair', 'close'])

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


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

    merged = today_df.join(prev_df, how='left')
    valid_previous = merged['prev_close'].notna() & merged['prev_close'].ne(0)
    merged['return_1d_pct'] = np.where(
        valid_previous,
        (merged['close'] / merged['prev_close'] - 1) * 100,
        np.nan,
    )
    return merged


def select_top_movers(df: pd.DataFrame, limit: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return strictly positive gainers and strictly negative losers."""
    valid = df.dropna(subset=['return_1d_pct'])
    gainers = valid[valid['return_1d_pct'] > 0].nlargest(limit, 'return_1d_pct')
    losers = valid[valid['return_1d_pct'] < 0].nsmallest(limit, 'return_1d_pct')
    return gainers, losers


# Local exchange timezone and regular-session open for each market.
_MARKET_SESSIONS = {
    'JKSE':  ('Asia/Jakarta', time(9, 0)),
    'SP500': ('America/New_York', time(9, 30)),
}


def latest_session_date(now: datetime, open_time: time) -> date:
    """Most recent weekday whose session has opened, in the market's local time."""
    session = now.date()
    if now.time() < open_time:
        session -= timedelta(days=1)
    while session.weekday() >= 5:
        session -= timedelta(days=1)
    return session


def is_session_complete(
    prices_df: pd.DataFrame,
    target_date: pd.Timestamp,
    min_ratio: float = 0.8,
) -> bool:
    """
    True when ``target_date`` has at least ``min_ratio`` of the symbols priced on
    the previous session, so a half-ingested day is not shown as a full market.
    """
    if prices_df.empty:
        return False
    counts = prices_df.groupby('date')['symbol'].nunique()
    if target_date not in counts.index:
        return False
    previous = counts[counts.index < target_date]
    if previous.empty:
        return True
    return counts[target_date] >= min_ratio * previous.iloc[-1]


_RETURN_THRESHOLDS = {
    '1D':  [-10, -5, -2, -0.5, 0.5, 2, 5, 10],
    '7D':  [-15, -10, -5, -2, 2, 5, 10, 15],
    '1M':  [-25, -15, -10, -5, 5, 10, 15, 25],
    '1Y':  [-50, -25, -10, -5, 5, 10, 25, 50],
    '10Y': [-100, -50, -25, -10, 10, 25, 50, 100],
}


def get_return_color_threshold(label: str) -> list[float] | None:
    """Resolve a horizon-aware scale for price and total-return variants."""
    normalized = label.removeprefix('Total ').replace(' USD', '')
    period = normalized.split()[0] if normalized else ''
    return _RETURN_THRESHOLDS.get(period)


def normalize_on_shared_date(
    prices: pd.DataFrame,
    group_col: str,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Rebase all selected series to 100 on their earliest shared valid date."""
    if prices.empty or group_col not in prices.columns:
        return pd.DataFrame(columns=list(prices.columns) + ['normalized']), None

    aligned = prices.copy()
    aligned['close'] = pd.to_numeric(aligned['close'], errors='coerce')
    aligned = aligned[
        aligned['close'].notna()
        & np.isfinite(aligned['close'])
        & aligned['close'].ne(0)
    ].drop_duplicates([group_col, 'date'], keep='last')
    if aligned.empty:
        return aligned.assign(normalized=pd.Series(dtype=float)), None

    group_count = aligned[group_col].nunique()
    shared_dates = (
        aligned.groupby('date')[group_col]
        .nunique()
        .loc[lambda counts: counts == group_count]
        .index
    )
    if shared_dates.empty:
        return aligned.iloc[0:0].assign(normalized=pd.Series(dtype=float)), None

    shared_start = pd.Timestamp(min(shared_dates))
    aligned = aligned[aligned['date'] >= shared_start].copy()
    baselines = aligned[aligned['date'] == shared_start].set_index(group_col)['close']
    aligned['normalized'] = aligned['close'] / aligned[group_col].map(baselines) * 100
    return aligned.sort_values([group_col, 'date']), shared_start


def color_legend_html(title: str, colors: list[str], breaks: list[float], unit: str) -> str:
    """Swatch legend for a binned treemap: one swatch per bin, left to right."""
    def fmt(value):
        return f'{value:g}{unit}'

    labels = (
        [f'< {fmt(breaks[0])}']
        + [f'{fmt(lo)} to {fmt(hi)}' for lo, hi in zip(breaks, breaks[1:])]
        + [f'> {fmt(breaks[-1])}']
    )
    items = ''.join(
        '<span style="display:inline-flex;align-items:center;gap:4px;'
        'margin-right:12px;white-space:nowrap">'
        f'<span style="width:12px;height:12px;border-radius:2px;background:{color};'
        'display:inline-block"></span>'
        f'{label}</span>'
        for color, label in zip(colors, labels)
    )
    return (
        '<div style="font-size:0.8rem;line-height:1.9;opacity:0.85">'
        f'<strong>{title}</strong>&nbsp;&nbsp;{items}</div>'
    )


def line_end_labels(norm_df: pd.DataFrame, group_col: str, color_scale: alt.Scale) -> alt.LayerChart:
    """Dot + name at each line's last point, so series are not told apart by colour alone."""
    last = norm_df.sort_values('date').groupby(group_col).tail(1)
    last = last.assign(label_y=dodge_label_positions(last['normalized'], norm_df['normalized']))
    dots = alt.Chart(last).mark_point(
        filled=True, size=60, opacity=1, stroke=cs.CHART_SURFACE, strokeWidth=2
    ).encode(
        x='date:T',
        y='normalized:Q',
        color=alt.Color(f'{group_col}:N', scale=color_scale, legend=None),
    )
    names = alt.Chart(last).mark_text(
        align='left', dx=8, fontSize=11, fontWeight=600, color=cs.CHART_TEXT
    ).encode(x='date:T', y='label_y:Q', text=f'{group_col}:N')
    return dots + names


def dodge_label_positions(
    end_values: pd.Series, all_values: pd.Series, min_gap_ratio: float = 0.07
) -> pd.Series:
    """
    Push line-end labels apart vertically so close finishes stay readable.
    The gap is a share of the plotted range, since pixel positions are unknown here.
    """
    span = float(all_values.max() - all_values.min()) or 1.0
    gap = span * min_gap_ratio
    ordered = end_values.sort_values()
    positions = []
    for value in ordered:
        positions.append(max(value, positions[-1] + gap) if positions else value)
    # Re-centre so the stack sits around the true end values, not only above them
    shift = (sum(positions) - float(ordered.sum())) / len(positions)
    return pd.Series([p - shift for p in positions], index=ordered.index).reindex(end_values.index)


def build_tree_input(
    df_tree: pd.DataFrame,
    size_col: str,
    size_var: str,
    color_col_data: pd.Series,
    color_var_label: str,
) -> pd.DataFrame:
    """
    Assemble the DataFrame passed to ``hd.prep_treemap``.

    ``size_var`` and ``color_var_label`` may be identical (e.g. the user picks
    'Dividend Yield' for both size and color).  In that case both series hold
    the same values and a single shared column is used — using the same label
    twice would collapse into one dict key, and the non-finite sanitization
    below would then fail with ``ValueError: Columns must be same length as key``.
    """
    cols = {
        'sector':   df_tree['sector'],
        'industry': df_tree['industry'],
        size_var:   df_tree[size_col],
    }
    if color_var_label != size_var:
        cols[color_var_label] = color_col_data

    tree_input = pd.DataFrame(cols, index=df_tree.index)

    # Sanitize non-finite values (inf / -inf) — these serialize to the literal
    # ``Infinity`` in JSON which the browser's JSON.parse rejects.  Convert
    # them to NaN so the subsequent dropna() cleans the offending rows.
    num_cols = list(dict.fromkeys([size_var, color_var_label]))
    tree_input[num_cols] = tree_input[num_cols].replace([np.inf, -np.inf], np.nan)
    return tree_input.dropna()


@st.cache_data(max_entries=256, ttl=60 * 60 * 4, show_spinner=False)
def get_usdidr_period_fx_factor(fmp_key: str, date_to_str: str, n_days: int) -> float | None:
    """
    Fetches USDIDR close rates for ``n_days`` before ``date_to_str`` and returns
    the IDR-to-USD appreciation factor for that window:

        fx_factor = usdidr_period_start / usdidr_period_end

    A value < 1.0 means IDR weakened over the period (hurts USD returns).
    Returns None if data is unavailable.
    """
    import requests as _req
    from datetime import date as _date, timedelta as _td
    date_from = (_date.fromisoformat(date_to_str) - _td(days=n_days + 14)).isoformat()
    try:
        url = (
            f'https://financialmodelingprep.com/api/v3/historical-price-full/USDIDR'
            f'?from={date_from}&to={date_to_str}&apikey={fmp_key}'
        )
        resp = _req.get(url, timeout=15)
        resp.raise_for_status()
        hist = resp.json().get('historical', [])
        if len(hist) < 2:
            return None
        history = pd.DataFrame(hist)
        history['date'] = pd.to_datetime(history['date'], errors='coerce')
        history['close'] = pd.to_numeric(history['close'], errors='coerce')
        history = history.dropna(subset=['date', 'close']).sort_values('date')
        requested_end = pd.Timestamp(date_to_str)
        end_rows = history[history['date'] <= requested_end]
        if end_rows.empty:
            return None
        actual_end = end_rows.iloc[-1]
        desired_start = actual_end['date'] - pd.Timedelta(days=n_days)
        start_rows = history[history['date'] <= desired_start]
        if start_rows.empty:
            return None
        usdidr_end = float(actual_end['close'])
        usdidr_start = float(start_rows.iloc[-1]['close'])
        if usdidr_end == 0:
            return None
        return usdidr_start / usdidr_end
    except Exception as e:
        logger.warning(f'Could not fetch USDIDR for {n_days}D window: {e}')
        return None


# ── URL query params — read once at the top ──────────────────────────────── #
# Supported params:
#   market  : 'JKSE' | 'SP500'
#   date    : 'YYYY-MM-DD'
#   size    : 'Market Cap' | 'Revenue' | 'Net Income' | 'Dividend Yield'
#   color   : any label from _COLOR_OPTION_COL_MAP
#   sector  : sector name | 'ALL'
#   group   : '1' | '0'
#   mcap    : integer (billions)

_qp = st.query_params

# ── Primary controls ──────────────────────────────────────────────────────── #

_MARKET_OPTIONS = ['Indonesian Stock (JKSE)', 'S&P 500 (US)']
_MARKET_SHORT_LABELS = {
    'Indonesian Stock (JKSE)': 'IDX',
    'S&P 500 (US)': 'S&P 500',
}
_market_qp      = _qp.get('market', '')
_market_default = (
    'S&P 500 (US)' if _market_qp.upper() == 'SP500'
    else 'Indonesian Stock (JKSE)'
)

def _market_code(market_label: str) -> str:
    return 'SP500' if market_label == 'S&P 500 (US)' else 'JKSE'


def _latest_trading_date(market_label: str) -> date:
    tz_name, open_time = _MARKET_SESSIONS[_market_code(market_label)]
    return latest_session_date(datetime.now(ZoneInfo(tz_name)), open_time)


def _show_latest_date():
    market = st.session_state.get('mw_sl', 'Indonesian Stock (JKSE)')
    st.session_state['mw_date'] = _latest_trading_date(market)


def _reset_heatmap_filters():
    market = st.session_state.get('mw_sl', 'Indonesian Stock (JKSE)')
    st.session_state['mw_date'] = _latest_trading_date(market)
    st.session_state['mw_size'] = 'Market Cap'
    st.session_state['mw_color'] = '1D Return %'
    st.session_state['mw_sector'] = 'ALL'
    st.session_state['mw_group'] = True
    st.session_state['mw_mcap'] = 1 if market == 'S&P 500 (US)' else 10


def _market_changed():
    market = st.session_state.get('mw_sl', 'Indonesian Stock (JKSE)')
    # The URL still names the market being left. If the user was on that
    # market's latest session, follow to the new market's latest session
    # (Jakarta and New York sessions fall on different calendar dates).
    previous = 'S&P 500 (US)' if st.query_params.get('market') == 'SP500' else 'Indonesian Stock (JKSE)'
    if st.session_state.get('mw_date') == _latest_trading_date(previous):
        st.session_state['mw_date'] = _latest_trading_date(market)
    st.session_state['mw_color'] = '1D Return %'
    st.session_state['mw_sector'] = 'ALL'
    st.session_state['mw_mcap'] = 1 if market == 'S&P 500 (US)' else 10


stock_select = toolbar.segmented_control(
    'Market',
    _MARKET_OPTIONS,
    default=_market_default,
    required=True,
    format_func=_MARKET_SHORT_LABELS.get,
    key='mw_sl',
    on_change=_market_changed,
    label_visibility='collapsed',
)

sl = _market_code(stock_select)
redis_key = 'div_score_jkse' if sl == 'JKSE' else 'div_score_sp500'

market_tz = ZoneInfo(_MARKET_SESSIONS[sl][0])
market_today = datetime.now(market_tz).date()
latest_trading_date = _latest_trading_date(stock_select)

# Parse date from URL param if present. This only seeds the widget; the
# Latest / Reset buttons always use ``latest_trading_date``.
initial_date = latest_trading_date
_date_qp = _qp.get('date', '')
try:
    _date_from_url = date.fromisoformat(_date_qp)
    if date(2015, 1, 1) <= _date_from_url <= market_today:
        initial_date = _date_from_url
except (ValueError, TypeError):
    pass

selected_date = toolbar.date_input(
    'Trading date',
    value=initial_date,
    max_value=market_today,
    min_value=date(2015, 1, 1),
    key='mw_date',
    label_visibility='collapsed',
    width=130,
)
toolbar.button(
    ':material/today:',
    help='Jump to the latest trading session',
    on_click=_show_latest_date,
)

# ── Load universe ─────────────────────────────────────────────────────────── #

universe_df = get_stock_universe(redis_key)

if universe_df.empty:
    st.error('Could not load stock universe. Please check the data pipeline.', icon='🚨')
    st.stop()

# Keep only needed columns (include multi-period return columns for color-by feature)
_KEEP = [c for c in [
    'sector', 'industry', 'mktCap', 'yield', 'medianProfitMargin',
    'revenueTTM', 'earningTTM', 'peRatio', 'psRatio', 'revenueGrowth',
    'return_7d', 'return_1m', 'return_1y', 'return_10y',
    'total_return_1y', 'total_return_10y',
] if c in universe_df.columns]
universe_df = universe_df[_KEEP].copy()
universe_df['mktCap_B'] = universe_df['mktCap'] / 1_000_000_000

# Convert fractional returns to percent (they are stored as 0-1 fractions)
_ret_pct_cols = ['return_7d', 'return_1m', 'return_1y', 'return_10y', 'total_return_1y', 'total_return_10y']
for _rc in _ret_pct_cols:
    if _rc in universe_df.columns:
        universe_df[_rc] = universe_df[_rc] * 100


# ── Treemap controls ──────────────────────────────────────────────────────── #

_SIZE_OPTIONS = ['Market Cap', 'Revenue', 'Net Income', 'Dividend Yield']
_size_qp      = _qp.get('size', '')
_size_default_idx = (
    _SIZE_OPTIONS.index(_size_qp)
    if _size_qp in _SIZE_OPTIONS else 0
)

size_var = toolbar.selectbox(
    'Tile size',
    options=_SIZE_OPTIONS,
    index=_size_default_idx,
    format_func=lambda value: f'Size: {value}',
    key='mw_size',
    label_visibility='collapsed',
    width=170,
)

# Build color-by options based on available columns.
# Sentinels starting with '__' are computed at render time, not from universe columns.
_COLOR_OPTION_COL_MAP = {
    '1D Return %':              None,               # % return (IDR) from stored / live data
    '7D Return %':              'return_7d',
    '7D USD Return %':          '__usd_return__',   # IDR return × FX factor, JKSE only
    '1M Return %':              'return_1m',
    '1M USD Return %':          '__usd_return__',
    '1Y Return %':              'return_1y',
    '1Y USD Return %':          '__usd_return__',
    '10Y Return %':             'return_10y',
    '10Y USD Return %':         '__usd_return__',
    'Total 1Y Return %':        'total_return_1y',
    'Total 1Y USD Return %':    '__usd_return__',
    'Total 10Y Return %':       'total_return_10y',
    'Total 10Y USD Return %':   '__usd_return__',
    'Dividend Yield':           'yield',
    'Profit Margin':            'medianProfitMargin',
    'Revenue Growth':           'revenueGrowth',
    'PE Ratio':                 'peRatio',
    'PS Ratio':                 'psRatio',
}

# Maps each USD return label → (idr_column, period_in_days)
_USD_RETURN_MAP = {
    '7D USD Return %':          ('return_7d',        7),
    '1M USD Return %':          ('return_1m',        30),
    '1Y USD Return %':          ('return_1y',        365),
    '10Y USD Return %':         ('return_10y',       3650),
    'Total 1Y USD Return %':    ('total_return_1y',  365),
    'Total 10Y USD Return %':   ('total_return_10y', 3650),
}

_available_color_opts = [
    k for k, v in _COLOR_OPTION_COL_MAP.items()
    if v is None
    or v in universe_df.columns
    # USD return options are only meaningful for IDR-denominated (JKSE) stocks
    or (v == '__usd_return__' and sl == 'JKSE' and _USD_RETURN_MAP.get(k, (None,))[0] in universe_df.columns)
]

_color_qp = _qp.get('color', '')
_color_default_idx = (
    _available_color_opts.index(_color_qp)
    if _color_qp in _available_color_opts else 0
)

color_var_label = toolbar.selectbox(
    'Tile color',
    options=_available_color_opts,
    index=_color_default_idx,
    format_func=lambda value: f'Color: {value}',
    key='mw_color',
    label_visibility='collapsed',
    width=205,
)

_sector_options = ['ALL'] + sorted(universe_df['sector'].dropna().unique().tolist())
_sector_qp      = _qp.get('sector', 'ALL')
_sector_default = _sector_qp if _sector_qp in _sector_options else 'ALL'

_group_qp  = _qp.get('group', '1')
_group_default = _group_qp != '0'

_mcap_default = 1 if sl == 'SP500' else 10
try:
    _mcap_qp = int(_qp.get('mcap', ''))
    if 0 <= _mcap_qp <= 1_000:
        _mcap_default = _mcap_qp
except (ValueError, TypeError):
    pass

# Count filters that differ from their defaults so the closed popover still
# tells people the map is narrowed.
_active_filters = sum([
    st.session_state.get('mw_sector', _sector_default) != 'ALL',
    not st.session_state.get('mw_group', _group_default),
    st.session_state.get('mw_mcap', _mcap_default) != (1 if sl == 'SP500' else 10),
])
with toolbar.popover(
    f':material/tune: {_active_filters}' if _active_filters else ':material/tune:',
    help='Filters: sector, minimum market cap, grouping',
):
    sector_filter = st.selectbox(
        'Sector',
        options=_sector_options,
        index=_sector_options.index(_sector_default),
        format_func=lambda value: 'All sectors' if value == 'ALL' else value,
        key='mw_sector',
    )
    market_cap_currency = 'IDR' if sl == 'JKSE' else 'USD'
    min_mcap_b = st.number_input(
        f'Minimum market cap ({market_cap_currency} billions)',
        min_value=0,
        max_value=1_000,
        value=_mcap_default,
        step=1,
        key='mw_mcap',
    )
    group_secs = st.toggle(
        'Group companies by sector', value=_group_default, key='mw_group'
    )
    st.button(
        'Reset filters',
        icon=':material/restart_alt:',
        on_click=_reset_heatmap_filters,
        width='stretch',
    )

# ── Sync widget state → URL query params ──────────────────────────────────── #
_qp['market']  = sl
_qp['date']    = selected_date.strftime('%Y-%m-%d')
_qp['size']    = size_var
_qp['color']   = color_var_label
_qp['sector']  = sector_filter
_qp['group']   = '1' if group_secs else '0'
_qp['mcap']    = str(min_mcap_b)

# ── Filter universe ───────────────────────────────────────────────────────── #

filtered_uni = universe_df[universe_df['mktCap_B'] >= min_mcap_b].copy()
if sector_filter != 'ALL':
    filtered_uni = filtered_uni[filtered_uni['sector'] == sector_filter]

# Drop GOOGL when GOOG is present to avoid duplicate Google entries in the treemap
if 'GOOG' in filtered_uni.index and 'GOOGL' in filtered_uni.index:
    filtered_uni = filtered_uni.drop(index='GOOGL')

symbols_tuple = tuple(sorted(filtered_uni.index.tolist()))

# ── Fetch prices for target date + a few days back ────────────────────────── #

selected_date_str = selected_date.strftime('%Y-%m-%d')
date_to_str   = selected_date_str
date_from_str = (selected_date - timedelta(days=14)).strftime('%Y-%m-%d')

with st.spinner(f'Fetching price data for {date_to_str}…'):
    prices_df = get_prices_for_date_range(sl.lower(), symbols_tuple, date_from_str, date_to_str)

target_ts = pd.Timestamp(selected_date)
returns_df = calc_daily_return_for_date(prices_df, target_ts)

# ── Fallback logic: FMP live profile → then previous stored date ──────────── #
# Priority order:
#   1. Stored prices for the selected date  (already computed above as returns_df)
#   2. FMP company profile live data        (when selected date is the latest opened session)
#   3. Most recent available stored date    (e.g. yesterday, for weekends / holidays)

_effective_date = selected_date
_data_source = 'Historical prices'
_as_of_label = selected_date_str
_price_notice = None


def _has_reported_returns(frame):
    return not frame.empty and frame['return_1d_pct'].notna().any()


def _is_usable_session(frame, ts, require_complete=True):
    return _has_reported_returns(frame) and (
        not require_complete or is_session_complete(prices_df, ts)
    )


if not _is_usable_session(returns_df, target_ts):
    _live_used = False
    # Live profiles carry the latest quote, i.e. the most recent session that
    # has opened in the market's own timezone. They must never stand in for an
    # older date, nor for "today" before the exchange has opened.
    if selected_date == latest_trading_date:
        with st.spinner('Fetching live price data from FMP…'):
            live_returns = get_live_returns_from_profile(symbols_tuple)
        if _has_reported_returns(live_returns):
            returns_df = live_returns
            _live_used = True
            _data_source = 'FMP live/latest company profiles'
            _as_of_label = datetime.now(market_tz).strftime('%Y-%m-%d %H:%M %Z')
            _price_notice = (
                f'Historical prices for **{selected_date_str}** are not complete yet. '
                f'Showing **live/latest** profile changes as of **{_as_of_label}**.'
            )

    # Try available historical dates newest-first, preferring fully ingested
    # sessions and only then accepting a partial one.
    if not _live_used and not prices_df.empty:
        available_dates = sorted(prices_df['date'].dropna().unique(), reverse=True)
        _found = False
        for require_complete in (True, False):
            for available_date in available_dates:
                candidate_date = pd.Timestamp(available_date).date()
                if candidate_date > selected_date:
                    continue
                candidate_ts = pd.Timestamp(candidate_date)
                candidate_returns = calc_daily_return_for_date(prices_df, candidate_ts)
                if not _is_usable_session(candidate_returns, candidate_ts, require_complete):
                    continue
                returns_df = candidate_returns
                _effective_date = candidate_date
                target_ts = candidate_ts
                date_to_str = candidate_date.strftime('%Y-%m-%d')
                _as_of_label = date_to_str
                if candidate_date == selected_date:
                    _price_notice = (
                        f'Prices for **{selected_date_str}** are still being loaded; '
                        'only part of the market is shown.'
                    )
                elif require_complete:
                    _price_notice = (
                        f'No complete session was available for **{selected_date_str}**. '
                        f'Showing the most recent complete trading date: **{date_to_str}**.'
                    )
                else:
                    _price_notice = (
                        f'No complete session was available for **{selected_date_str}**. '
                        f'Showing the most recent partial trading date: **{date_to_str}**.'
                    )
                _found = True
                break
            if _found:
                break

# ── KPI row ───────────────────────────────────────────────────────────────── #

if not _has_reported_returns(returns_df):
    st.warning(
        f'No usable price changes were found for **{selected_date_str}**. '
        'Try the latest session, another trading date, or a broader filter.',
        icon='⚠️'
    )
else:
    valid_returns = returns_df['return_1d_pct'].dropna()
    n_reporting = int(valid_returns.size)
    n_unavailable = max(len(symbols_tuple) - n_reporting, 0)
    coverage_pct = n_reporting / len(symbols_tuple) * 100 if symbols_tuple else 0
    n_gainers = int((valid_returns > 0).sum())
    n_losers = int((valid_returns < 0).sum())
    n_flat = int((valid_returns == 0).sum())
    avg_return = float(valid_returns.mean())
    med_return = float(valid_returns.median())
    breadth_pct = n_gainers / n_reporting * 100 if n_reporting else 0

    # Details that used to be separate notices go into one tooltip
    _status_details = [
        f'Effective trading date: {date_to_str}',
        f'Source: {_data_source}',
        f'As of: {_as_of_label}',
        f'{n_flat} unchanged · {n_unavailable} without a price change',
    ]
    if _price_notice:
        _status_details.insert(0, _price_notice)

    # ── Merge returns into universe ───────────────────────────────────────── #

    df_tree = filtered_uni.copy()
    _ret_cols = [c for c in ['return_1d_pct'] if c in returns_df.columns]
    df_tree = df_tree.join(returns_df[_ret_cols], how='left')

    if _effective_date != latest_trading_date and color_var_label != '1D Return %':
        _status_details.append(
            f'{color_var_label} uses the latest universe snapshot; only 1D Return % '
            f'is calculated for the effective trading date {date_to_str}.'
        )

    # ── If a USD return variant is requested, fetch the period FX factor ───── #
    _fx_factor = None
    if color_var_label in _USD_RETURN_MAP:
        _idr_col, _n_days = _USD_RETURN_MAP[color_var_label]
        # The stored IDR returns come from the latest universe snapshot, so the
        # FX window must end on the latest session too, not the selected date.
        with st.spinner(f'Fetching USDIDR rate for {color_var_label}…'):
            _fx_factor = get_usdidr_period_fx_factor(
                api_key, latest_trading_date.isoformat(), _n_days
            )
        if _fx_factor is None:
            st.toast(
                f'Could not fetch the USDIDR rate for {color_var_label}; '
                'showing the IDR return instead.',
                icon=':material/warning:',
            )
            # Fall back to the corresponding IDR variant
            color_var_label = color_var_label.replace(' USD', '')

    # Map size variable
    size_col_map = {
        'Market Cap':      'mktCap_B',
        'Revenue':         'revenueTTM',
        'Net Income':      'earningTTM',
        'Dividend Yield':  'yield',
    }
    size_col = size_col_map[size_var]

    # Drop rows where size column is missing or <= 0
    df_tree = df_tree[df_tree[size_col].notna() & (df_tree[size_col] > 0)]

    # ── Resolve color column ───────────────────────────────────────────────── #
    _color_src_col = _COLOR_OPTION_COL_MAP[color_var_label]
    if _color_src_col is None:
        # '1D Return %' — IDR % change from live/stored data
        color_col_data = df_tree['return_1d_pct']
    elif _color_src_col == '__usd_return__':
        # Multi-period USD return — apply FX factor to the stored IDR return column
        # Formula: usd_return = (1 + idr_return/100) × (usdidr_start / usdidr_end) − 1
        #          fx_factor   = usdidr_start / usdidr_end  (< 1 when IDR weakened)
        _idr_col, _ = _USD_RETURN_MAP[color_var_label]
        idr_col_data = df_tree[_idr_col] if _idr_col in df_tree.columns else df_tree['return_1d_pct']
        color_col_data = ((1 + idr_col_data / 100) * _fx_factor - 1) * 100
    elif _color_src_col in df_tree.columns:
        color_col_data = df_tree[_color_src_col]
    else:
        # Fallback: column not available — revert to 1D %
        color_col_data = df_tree['return_1d_pct']
        color_var_label = '1D Return %'

    tree_input = build_tree_input(
        df_tree,
        size_col=size_col,
        size_var=size_var,
        color_col_data=color_col_data,
        color_var_label=color_var_label,
    )

    # ── Color map & threshold based on selected variable ─────────────────── #
    _return_labels = {
        '1D Return %',
        '7D Return %',        '7D USD Return %',
        '1M Return %',        '1M USD Return %',
        '1Y Return %',        '1Y USD Return %',
        '10Y Return %',       '10Y USD Return %',
        'Total 1Y Return %',  'Total 1Y USD Return %',
        'Total 10Y Return %', 'Total 10Y USD Return %',
    }
    if color_var_label in _return_labels:
        color_map = 'red_green'
        color_threshold = get_return_color_threshold(color_var_label)
    elif color_var_label == 'PE Ratio':
        color_map = 'red_shade'
        color_threshold = [-100, 0, 5, 15]
    elif color_var_label == 'PS Ratio':
        color_map = 'red_shade'
        color_threshold = [-1000, -100, -10, -1, 0, 1, 2, 3, 5]
    elif color_var_label == 'Dividend Yield':
        color_map = 'dividend_green'
        color_threshold = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        color_map = 'green_shade'
        color_threshold = None

    # ── Status line ───────────────────────────────────────────────────────── #

    if _data_source.startswith('FMP'):
        _status_badge = ':blue-badge[:material/sensors: Live]'
    elif _price_notice:
        _status_badge = f':orange-badge[:material/info: Showing {date_to_str}]'
    else:
        _status_badge = f':gray-badge[{date_to_str}]'
    with stats_slot:
        st.caption(
            f'{_status_badge}&ensp;**{breadth_pct:.0f}%** of stocks up '
            f'(:green[▲ {n_gainers}] :red[▼ {n_losers}])'
            f'&ensp;·&ensp;Median **{med_return:+.2f}%**'
            f'&ensp;·&ensp;Equal-weight **{avg_return:+.2f}%**'
            f'&ensp;·&ensp;{n_reporting} of {len(symbols_tuple)} stocks priced',
            help='  \n'.join(_status_details),
        )

    # ── Build treemap ─────────────────────────────────────────────────────── #

    if tree_input.empty:
        st.info(
            'No companies have both a tile size and a tile color for this selection. '
            'Try another measure or a broader filter.',
            icon=':material/filter_alt_off:',
        )
    else:
        tree_data = hd.prep_treemap(
            tree_input,
            size_var=size_var,
            color_var=color_var_label,
            color_threshold=color_threshold,
            add_label='color_var',
            group_secs=group_secs,
        )

        option = hp.plot_treemap(
            tree_data,
            size_var=size_var,
            color_var=color_var_label,
            show_gradient=True,
            colormap=color_map,
            group_secs=group_secs,
        )

        # Edge-to-edge map: the toolbar already names the measures, and the
        # root "ALL" header row adds nothing
        option['title'] = {'show': False}
        _series = option['series'][0]
        _series.pop('width', None)
        _series.pop('height', None)
        _series.update({'top': 0, 'left': 0, 'right': 0, 'bottom': 28})
        _series['levels'][0]['upperLabel'] = {'show': False}

        target_market = 'JKSE' if sl == 'JKSE' else 'S&P500'
        click_event_js = (
            "function(params){"
            "if(params.data.children){return null;}"
            "var symbol=String(params.name).split('\\n')[0];"
            "var origin=document.referrer?new URL(document.referrer).origin:window.location.origin;"
            f"var url=origin+'/stock_picker?market={target_market}&stock='+encodeURIComponent(symbol);"
            "window.open(url,'_blank','noopener,noreferrer');"
            "return null;"
            "}"
        )

        # Fill the rest of the first screen, leaving room for the legend below
        treemap_height = remaining_viewport_height('mw_viewport', reserve=64)
        st_echarts(
            option,
            events={'click': click_event_js},
            height=f'{treemap_height}px',
            width='100%',
            key='mw_treemap',
        )

        # Same breakpoints prep_treemap bins by: fixed thresholds or quartiles
        legend_breaks = color_threshold or [
            float(v) for v in np.percentile(tree_input[color_var_label], [25, 50, 75])
        ]
        st.html(color_legend_html(
            color_var_label,
            hp.treemap_bin_colors(color_map, len(legend_breaks) + 1),
            legend_breaks,
            unit='' if 'Ratio' in color_var_label else '%',
        ))

    # ── Distribution chart ────────────────────────────────────────────────── #

    with st.expander('Return distribution', expanded=False):
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

            clipped = dist_df[(dist_df['return'] >= p2) & (dist_df['return'] <= p98)]['return']
            excluded_count = len(dist_df) - len(clipped)

            counts, bin_edges = np.histogram(clipped, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist_df = pd.DataFrame({
                'return': bin_centers,
                'count':  counts,
                'color':  [cs.CHART_POSITIVE if c >= 0 else cs.CHART_NEGATIVE for c in bin_centers],
            })

            zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                color=cs.CHART_NEUTRAL, strokeWidth=1.5, strokeDash=[4, 3], opacity=0.8
            ).encode(x=alt.X('x:Q', scale=alt.Scale(domain=[p2, p98])))

            hist = (
                alt.Chart(hist_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X('return:Q', title='1D Return (%)',
                            scale=alt.Scale(domain=[p2, p98])),
                    y=alt.Y('count:Q', title='Number of stocks'),
                    color=alt.Color('color:N', scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip('return:Q', title='Return (%)', format='.2f'),
                        alt.Tooltip('count:Q',  title='Number of stocks'),
                    ],
                )
                .properties(height=260)
            )

            st.altair_chart((hist + zero_line), width='stretch', theme=None)
            if excluded_count:
                st.caption(
                    f'{excluded_count} extreme observation(s) outside the 2nd–98th '
                    'percentile range are excluded from this histogram only.'
                )

    # ── Top gainers / losers table ────────────────────────────────────────── #

    with st.expander('Top gainers and losers', expanded=False):
        merged_tbl = filtered_uni[['sector', 'industry', 'mktCap_B']].join(
            returns_df[['close', 'prev_close', 'return_1d_pct']], how='inner'
        ).dropna(subset=['return_1d_pct'])

        col_g, col_l = st.columns(2)

        top_gainers, top_losers = select_top_movers(merged_tbl)
        top_gainers = top_gainers.reset_index(names='stock')
        top_losers = top_losers.reset_index(names='stock')

        cfig_g = {
            'stock': st.column_config.TextColumn('Stock'),
            'sector': st.column_config.TextColumn('Sector'),
            'close': st.column_config.NumberColumn('Close', format='%,.2f'),
            'prev_close': st.column_config.NumberColumn('Previous close', format='%,.2f'),
            'return_1d_pct': st.column_config.NumberColumn('1D Return %', format='%+.2f%%'),
            'mktCap_B': st.column_config.NumberColumn(
                f'Market cap ({market_cap_currency} bn)', format='%,.1f'
            ),
            'industry': None,
        }

        with col_g:
            st.markdown(f'#### Gainers ({len(top_gainers)})')
            st.dataframe(
                top_gainers[['stock', 'sector', 'close', 'prev_close', 'return_1d_pct', 'mktCap_B']],
                column_config=cfig_g,
                hide_index=True,
            )

        with col_l:
            st.markdown(f'#### Losers ({len(top_losers)})')
            st.dataframe(
                top_losers[['stock', 'sector', 'close', 'prev_close', 'return_1d_pct', 'mktCap_B']],
                column_config=cfig_g,
                hide_index=True,
            )

    load_macro_context = st.toggle(
        'Load global index and currency analysis',
        value=False,
        key='mw_load_macro',
        help=(
            'These comparisons use additional external requests. Keep this off '
            'when you only need the stock heatmap.'
        ),
    )
    if not load_macro_context:
        st.caption(
            'Global indices and currency pairs are loaded on demand to keep the '
            'primary heatmap responsive.'
        )
        st.stop()

    # ── Global Index Comparison ───────────────────────────────────────────── #

    st.divider()
    st.subheader('Global index comparison')
    st.caption('Normalized performance (rebased to 100) vs IHSG — select a time window to compare.')

    _PERIOD_OPTIONS = {
        '1W':   7,
        '1M':   30,
        '3M':   90,
        '6M':   180,
        'YTD':  None,   # handled specially
        '1Y':   365,
        '2Y':   730,
        '3Y':   1095,
        '5Y':   1825,
        '10Y':  3650,
    }

    idx_col1, idx_col2 = st.columns([3, 1])
    with idx_col2:
        period_label = st.selectbox(
            'Period',
            options=list(_PERIOD_OPTIONS.keys()),
            index=2,          # default 3M
            key='mw_idx_period',
        )
        visible_indices = st.multiselect(
            'Show indices',
            options=list(_INDEX_SYMBOLS.values()),
            default=['IHSG', 'S&P 500', 'Nikkei 225'],
            key='mw_idx_visible',
        )

    # Compute date range for index fetch
    idx_to   = _effective_date
    if period_label == 'YTD':
        idx_from = date(idx_to.year, 1, 1)
    else:
        idx_from = idx_to - timedelta(days=_PERIOD_OPTIONS[period_label])

    idx_from_str = idx_from.strftime('%Y-%m-%d')
    idx_to_str   = idx_to.strftime('%Y-%m-%d')

    selected_index_symbols = tuple(
        symbol for symbol, label in _INDEX_SYMBOLS.items() if label in visible_indices
    )
    idx_prices_df = get_index_prices(
        api_key, idx_from_str, idx_to_str, selected_index_symbols
    )

    with idx_col1:
        if idx_prices_df.empty:
            st.warning('Could not load index price data. Check FMP API key or network.', icon='⚠️')
        else:
            # Filter to selected indices
            idx_prices_df = idx_prices_df[idx_prices_df['index'].isin(visible_indices)]

            if idx_prices_df.empty:
                st.info('Select at least one index to display.')
            else:
                norm_df, idx_shared_start = normalize_on_shared_date(
                    idx_prices_df, group_col='index'
                )
                idx_prices_df = norm_df
                if idx_shared_start is not None:
                    st.caption(
                        f'All selected indices are rebased from their first shared '
                        f'trading date: {idx_shared_start.date()}.'
                    )

                # Colour scale
                domain_idx    = [lbl for lbl in _INDEX_COLORS if lbl in visible_indices]
                range_idx     = [_INDEX_COLORS[lbl] for lbl in domain_idx]

                # Highlight IHSG with a thicker, full-opacity line
                norm_others = norm_df[norm_df['index'] != 'IHSG']
                norm_ihsg   = norm_df[norm_df['index'] == 'IHSG']

                sel = alt.selection_point(fields=['index'], bind='legend')

                base_line = alt.Chart(norm_others).mark_line(
                    strokeWidth=1.8,
                    opacity=0.75,
                    interpolate='monotone',
                ).encode(
                    x=alt.X('date:T', title=''),
                    y=alt.Y('normalized:Q', title='Indexed (100 = period start)',
                            scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'index:N',
                        scale=alt.Scale(domain=domain_idx, range=range_idx),
                        legend=alt.Legend(title='Index', orient='bottom'),
                    ),
                    opacity=alt.condition(sel, alt.value(0.85), alt.value(0.15)),
                    tooltip=[
                        alt.Tooltip('date:T',         title='Date'),
                        alt.Tooltip('index:N',        title='Index'),
                        alt.Tooltip('normalized:Q',   title='Indexed',  format='.2f'),
                        alt.Tooltip('close:Q',        title='Close',    format=',.2f'),
                    ],
                ).add_params(sel)

                ihsg_line = alt.Chart(norm_ihsg).mark_line(
                    strokeWidth=3.5,
                    opacity=1.0,
                    interpolate='monotone',
                    strokeDash=[],
                ).encode(
                    x=alt.X('date:T'),
                    y=alt.Y('normalized:Q', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'index:N',
                        scale=alt.Scale(domain=domain_idx, range=range_idx),
                    ),
                    tooltip=[
                        alt.Tooltip('date:T',         title='Date'),
                        alt.Tooltip('index:N',        title='Index'),
                        alt.Tooltip('normalized:Q',   title='Indexed',  format='.2f'),
                        alt.Tooltip('close:Q',        title='Close',    format=',.2f'),
                    ],
                )

                baseline_rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
                    strokeDash=[4, 3], color=cs.CHART_NEUTRAL, strokeWidth=1, opacity=0.7
                ).encode(y='y:Q')

                perf_chart = (
                    alt.layer(
                        baseline_rule, base_line, ihsg_line,
                        line_end_labels(
                            norm_df, 'index',
                            alt.Scale(domain=domain_idx, range=range_idx),
                        ),
                    )
                    .properties(height=360)
                    .resolve_scale(color='shared')
                )
                st.altair_chart(perf_chart, width='stretch', theme=None)

    # ── Index metric cards (1D return + period return) ───────────────────────#

    if not idx_prices_df.empty:
        # Fetch a 1-week window around selected_date to compute the 1D return
        all_idx_full = get_index_prices(
            api_key,
            (idx_to - timedelta(days=7)).strftime('%Y-%m-%d'),
            idx_to_str,
            selected_index_symbols,
        )

        if not all_idx_full.empty:
            all_idx_full   = all_idx_full[all_idx_full['index'].isin(visible_indices)]
            target_ts_idx  = pd.Timestamp(idx_to)

            # Also compute period return from the already-fetched norm_df
            # norm_df is in scope from the block above (inside idx_col1 `with`).
            # Re-compute it safely here using idx_prices_df (already filtered).
            def _period_return(g):
                g = g.sort_values('date')
                if len(g) < 2:
                    return None
                return (g['close'].iloc[-1] / g['close'].iloc[0] - 1) * 100

            period_rets = (
                idx_prices_df
                .groupby('index')
                .apply(_period_return, include_groups=False)
                .dropna()
            )  # Series: index → period_return_pct

            def _get_1d_return(g):
                g = g.sort_values('date')
                today_row  = g[g['date'] == target_ts_idx]
                prev_rows  = g[g['date'] < target_ts_idx]
                if today_row.empty or prev_rows.empty:
                    return None
                close      = float(today_row['close'].iloc[0])
                prev_close = float(prev_rows.iloc[-1]['close'])
                return (close / prev_close - 1) * 100 if prev_close else None

            daily_rets = (
                all_idx_full
                .groupby('index')
                .apply(_get_1d_return, include_groups=False)
                .dropna()
            )  # Series: index → 1d_return_pct

            # Build sorted order: IHSG first, then descending by period return
            all_labels = [lbl for lbl in _INDEX_SYMBOLS.values() if lbl in visible_indices]
            sorted_labels = sorted(
                all_labels,
                key=lambda lbl: (lbl != 'IHSG', -(period_rets.get(lbl, float('-inf')))),
            )

            st.markdown(
                f'**Index returns over {period_label}, with 1D change**'
            )
            # Render cards in rows of 5
            cards_per_row = 5
            for row_start in range(0, len(sorted_labels), cards_per_row):
                row_labels = sorted_labels[row_start: row_start + cards_per_row]
                cols = st.columns(len(row_labels))
                for col, lbl in zip(cols, row_labels):
                    d_ret  = daily_rets.get(lbl)
                    p_ret  = period_rets.get(lbl)
                    p_str  = f'{p_ret:+.2f}%'  if p_ret  is not None else 'N/A'
                    # Signed delta string so Streamlit picks arrow and colour from it
                    d_str  = f'{d_ret:+.2f}% 1D' if d_ret is not None else None
                    flag    = _INDEX_FLAGS.get(lbl, '')
                    col.metric(
                        label=f'{flag} {lbl}',
                        value=p_str,
                        delta=d_str,
                        help=f'Period return over **{period_label}** | 1-day return on {date_to_str}',
                    )

    # ── Currency Watch (Forex vs IDR) ─────────────────────────────────────── #

    st.divider()
    st.subheader('Currency watch vs Indonesian rupiah')
    st.caption('Exchange rates of major currencies against IDR — normalized to 100 at the start of the selected period.')

    fx_col1, fx_col2 = st.columns([3, 1])
    with fx_col2:
        fx_period_label = st.selectbox(
            'Period',
            options=list(_PERIOD_OPTIONS.keys()),
            index=2,          # default 3M
            key='mw_fx_period',
        )
        all_fx_labels = [v[0] for v in _FX_SYMBOLS.values()]
        visible_pairs = st.multiselect(
            'Show pairs',
            options=all_fx_labels,
            default=['USD/IDR', 'EUR/IDR', 'SGD/IDR'],
            key='mw_fx_visible',
        )

    # Compute date range for FX fetch
    fx_to = _effective_date
    if fx_period_label == 'YTD':
        fx_from = date(fx_to.year, 1, 1)
    else:
        fx_from = fx_to - timedelta(days=_PERIOD_OPTIONS[fx_period_label])

    fx_from_str = fx_from.strftime('%Y-%m-%d')
    fx_to_str   = fx_to.strftime('%Y-%m-%d')

    selected_fx_symbols = tuple(
        symbol for symbol, label in _FX_LABELS.items() if label in visible_pairs
    )
    fx_prices_df = get_fx_prices(
        api_key, fx_from_str, fx_to_str, selected_fx_symbols
    )

    with fx_col1:
        if fx_prices_df.empty:
            st.warning('Could not load currency rate data. Check FMP API key or network.', icon='⚠️')
        else:
            fx_filtered = fx_prices_df[fx_prices_df['pair'].isin(visible_pairs)]

            if fx_filtered.empty:
                st.info('Select at least one currency pair to display.')
            else:
                fx_norm_df, fx_shared_start = normalize_on_shared_date(
                    fx_filtered, group_col='pair'
                )
                if fx_shared_start is not None:
                    st.caption(
                        f'All selected pairs are rebased from their first shared '
                        f'trading date: {fx_shared_start.date()}.'
                    )

                # Build color scale from label → color
                fx_domain = [v[0] for v in _FX_SYMBOLS.values() if v[0] in visible_pairs]
                fx_range  = [v[2] for v in _FX_SYMBOLS.values() if v[0] in visible_pairs]

                fx_sel = alt.selection_point(fields=['pair'], bind='legend')

                fx_line = alt.Chart(fx_norm_df).mark_line(
                    strokeWidth=2.0,
                    interpolate='monotone',
                ).encode(
                    x=alt.X('date:T', title=''),
                    y=alt.Y('normalized:Q', title='Indexed (100 = period start)',
                            scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'pair:N',
                        scale=alt.Scale(domain=fx_domain, range=fx_range),
                        legend=alt.Legend(title='Currency pair', orient='bottom'),
                    ),
                    opacity=alt.condition(fx_sel, alt.value(0.9), alt.value(0.15)),
                    tooltip=[
                        alt.Tooltip('date:T',       title='Date'),
                        alt.Tooltip('pair:N',       title='Pair'),
                        alt.Tooltip('normalized:Q', title='Indexed',      format='.2f'),
                        alt.Tooltip('close:Q',      title='Rate (IDR)',   format=',.2f'),
                    ],
                ).add_params(fx_sel)

                fx_baseline = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
                    strokeDash=[4, 3], color=cs.CHART_NEUTRAL, strokeWidth=1, opacity=0.7
                ).encode(y='y:Q')

                fx_chart = (
                    alt.layer(
                        fx_baseline, fx_line,
                        line_end_labels(
                            fx_norm_df, 'pair',
                            alt.Scale(domain=fx_domain, range=fx_range),
                        ),
                    )
                    .properties(height=360)
                    .resolve_scale(color='shared')
                )
                st.altair_chart(fx_chart, width='stretch', theme=None)

    # ── FX metric cards (1D return + period return) ──────────────────────── #

    if not fx_prices_df.empty and visible_pairs:
        # Fetch a short window around selected_date for 1D return calculation
        fx_full = get_fx_prices(
            api_key,
            (fx_to - timedelta(days=7)).strftime('%Y-%m-%d'),
            fx_to_str,
            selected_fx_symbols,
        )

        if not fx_full.empty:
            fx_full = fx_full[fx_full['pair'].isin(visible_pairs)]
            target_ts_fx = pd.Timestamp(fx_to)

            def _fx_period_return(g):
                g = g.sort_values('date')
                if len(g) < 2:
                    return None
                return (g['close'].iloc[-1] / g['close'].iloc[0] - 1) * 100

            fx_period_df = fx_prices_df[fx_prices_df['pair'].isin(visible_pairs)]
            fx_period_rets = (
                fx_period_df
                .groupby('pair')
                .apply(_fx_period_return, include_groups=False)
                .dropna()
            )

            def _fx_1d_return(g):
                g = g.sort_values('date')
                today_row = g[g['date'] == target_ts_fx]
                prev_rows = g[g['date'] < target_ts_fx]
                if today_row.empty or prev_rows.empty:
                    return None
                close     = float(today_row['close'].iloc[0])
                prev_close = float(prev_rows.iloc[-1]['close'])
                return (close / prev_close - 1) * 100 if prev_close else None

            fx_daily_rets = (
                fx_full
                .groupby('pair')
                .apply(_fx_1d_return, include_groups=False)
                .dropna()
            )

            # Sort by period return descending
            sorted_pairs = sorted(
                visible_pairs,
                key=lambda lbl: -(fx_period_rets.get(lbl, float('-inf'))),
            )

            st.markdown(f'**Currency returns over {fx_period_label}, with 1D change**')
            fx_cards_per_row = 5
            for row_start in range(0, len(sorted_pairs), fx_cards_per_row):
                row_pairs = sorted_pairs[row_start: row_start + fx_cards_per_row]
                fx_cols = st.columns(len(row_pairs))
                for col, lbl in zip(fx_cols, row_pairs):
                    sym = _FX_LABEL_TO_SYM.get(lbl, '')
                    flag   = _FX_FLAGS.get(sym, '')
                    d_ret  = fx_daily_rets.get(lbl)
                    p_ret  = fx_period_rets.get(lbl)
                    p_str  = f'{p_ret:+.2f}%' if p_ret is not None else 'N/A'
                    d_str  = f'{d_ret:+.2f}% 1D' if d_ret is not None else None
                    col.metric(
                        label=f'{flag} {lbl}',
                        value=p_str,
                        delta=d_str,
                        help=f'IDR per 1 unit of foreign currency — period return over **{fx_period_label}** | 1-day return on {date_to_str}',
                    )
